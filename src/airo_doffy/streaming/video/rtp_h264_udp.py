"""Experimental RFC 6184 H.264 RTP packetization over UDP."""

from __future__ import annotations

import secrets
import socket
import struct
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ...config.models import NetworkConfig, VideoStreamingConfig
from ...core.buffers import is_newer_sequence
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import EncodedFrame, VideoCodec

RTP_HEADER = struct.Struct("!BBHII")
RTP_HEADER_SIZE = RTP_HEADER.size
RTP_CLOCK_HZ = 90_000
H264_FU_A_TYPE = 28


class DatagramSocket(Protocol):
    def sendto(self, data: bytes, target: tuple[str, int]) -> int:
        """Send one datagram."""

    def close(self) -> None:
        """Release the socket."""


SocketFactory = Callable[[], DatagramSocket]


@dataclass(frozen=True, slots=True)
class ParsedRtpPacket:
    """Fields used by the bounded experimental jitter buffer."""

    marker: bool
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes


def split_h264_access_unit(
    access_unit: bytes | bytearray | memoryview,
) -> tuple[bytes, ...]:
    """Split Annex B, four-byte AVCC, or one raw NAL access unit."""

    try:
        data = bytes(access_unit)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("H.264 access unit must support bytes") from exc
    if not data:
        raise ModelValidationError("H.264 access unit must not be empty")

    starts: list[tuple[int, int]] = []
    index = 0
    while index <= len(data) - 3:
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    if starts:
        nal_units = tuple(
            data[position + prefix : starts[item + 1][0]]
            if item + 1 < len(starts)
            else data[position + prefix :]
            for item, (position, prefix) in enumerate(starts)
        )
    else:
        avcc_units: list[bytes] = []
        offset = 0
        while offset + 4 <= len(data):
            length = int.from_bytes(data[offset : offset + 4], "big")
            if length == 0 or offset + 4 + length > len(data):
                avcc_units = []
                break
            avcc_units.append(data[offset + 4 : offset + 4 + length])
            offset += 4 + length
        nal_units = (
            tuple(avcc_units)
            if avcc_units and offset == len(data)
            else (data,)
        )

    if not nal_units or any(not nal for nal in nal_units):
        raise ModelValidationError("H.264 access unit contains an empty NAL")
    for nal in nal_units:
        nal_type = nal[0] & 0x1F
        if not 1 <= nal_type <= 23:
            raise ModelValidationError(
                f"unsupported H.264 NAL unit type: {nal_type}"
            )
    return nal_units


def packetize_h264_rtp(
    access_unit: bytes | bytearray | memoryview,
    *,
    sequence_start: int,
    timestamp: int,
    ssrc: int,
    payload_type: int = 96,
    mtu: int = 1200,
) -> tuple[bytes, ...]:
    """Packetize one H.264 access unit using single NAL or FU-A packets."""

    for value, maximum, name in (
        (sequence_start, 0xFFFF, "sequence_start"),
        (timestamp, 0xFFFFFFFF, "timestamp"),
        (ssrc, 0xFFFFFFFF, "ssrc"),
        (payload_type, 127, "payload_type"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= maximum
        ):
            raise ModelValidationError(f"{name} must be within [0, {maximum}]")
    if isinstance(mtu, bool) or not isinstance(mtu, int) or mtu < 256:
        raise ModelValidationError("RTP MTU must be an integer >= 256")
    if mtu > 65_507:
        raise ModelValidationError("RTP MTU must be <= 65507")

    nal_units = split_h264_access_unit(access_unit)
    max_payload = mtu - RTP_HEADER_SIZE
    packets: list[tuple[bytes, bool]] = []
    for nal_index, nal in enumerate(nal_units):
        final_nal = nal_index == len(nal_units) - 1
        if len(nal) <= max_payload:
            packets.append((nal, final_nal))
            continue
        fragment_size = max_payload - 2
        nal_header = nal[0]
        fu_indicator = (nal_header & 0xE0) | H264_FU_A_TYPE
        body = nal[1:]
        fragment_count = (len(body) + fragment_size - 1) // fragment_size
        for fragment_index in range(fragment_count):
            start = fragment_index * fragment_size
            end = min(start + fragment_size, len(body))
            fu_header = nal_header & 0x1F
            if fragment_index == 0:
                fu_header |= 0x80
            if fragment_index == fragment_count - 1:
                fu_header |= 0x40
            marker = final_nal and fragment_index == fragment_count - 1
            packets.append((bytes((fu_indicator, fu_header)) + body[start:end], marker))

    return tuple(
        RTP_HEADER.pack(
            0x80,
            payload_type | (0x80 if marker else 0),
            (sequence_start + index) & 0xFFFF,
            timestamp,
            ssrc,
        )
        + payload
        for index, (payload, marker) in enumerate(packets)
    )


def parse_rtp_packet(packet: bytes | bytearray | memoryview) -> ParsedRtpPacket:
    """Parse one RTP v2 packet, including CSRC/extension/padding offsets."""

    try:
        data = bytes(packet)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("RTP packet must support bytes") from exc
    if len(data) < RTP_HEADER_SIZE:
        raise ModelValidationError("RTP packet is truncated")
    first, second, sequence, timestamp, ssrc = RTP_HEADER.unpack_from(data)
    if first >> 6 != 2:
        raise ModelValidationError("RTP version must be 2")
    csrc_count = first & 0x0F
    offset = RTP_HEADER_SIZE + csrc_count * 4
    if offset > len(data):
        raise ModelValidationError("RTP CSRC list is truncated")
    if first & 0x10:
        if offset + 4 > len(data):
            raise ModelValidationError("RTP extension header is truncated")
        extension_words = int.from_bytes(data[offset + 2 : offset + 4], "big")
        offset += 4 + extension_words * 4
        if offset > len(data):
            raise ModelValidationError("RTP extension payload is truncated")
    payload_end = len(data)
    if first & 0x20:
        padding = data[-1]
        if padding == 0 or padding > payload_end - offset:
            raise ModelValidationError("RTP padding is invalid")
        payload_end -= padding
    payload = data[offset:payload_end]
    if not payload:
        raise ModelValidationError("RTP payload must not be empty")
    return ParsedRtpPacket(
        marker=bool(second & 0x80),
        payload_type=second & 0x7F,
        sequence=sequence,
        timestamp=timestamp,
        ssrc=ssrc,
        payload=payload,
    )


@dataclass(frozen=True, slots=True)
class RtpH264TransportMetrics:
    frames_sent: int
    packets_sent: int
    bytes_sent: int
    dropped_stale: int
    dropped_late: int
    errors: int


def _udp_socket() -> DatagramSocket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class RtpH264UdpTransport:
    """One-stream H.264 RTP sender with stale and late frame dropping."""

    def __init__(
        self,
        target_host: str,
        target_port: int,
        config: VideoStreamingConfig,
        *,
        ssrc: int | None = None,
        initial_sequence: int | None = None,
        initial_timestamp: int | None = None,
        clock: Clock | None = None,
        socket_factory: SocketFactory = _udp_socket,
    ) -> None:
        if not isinstance(target_host, str) or not target_host.strip():
            raise ModelValidationError("target_host must be a non-empty string")
        if (
            isinstance(target_port, bool)
            or not isinstance(target_port, int)
            or not 1 <= target_port <= 65535
        ):
            raise ModelValidationError("target_port must be within [1, 65535]")
        self._target = (target_host, target_port)
        self._mtu = config.rtp_mtu
        self._payload_type = config.rtp_payload_type
        self._max_age_ns = round(config.max_frame_age_s * 1_000_000_000)
        self._ssrc = secrets.randbits(32) if ssrc is None else ssrc
        self._sequence = (
            secrets.randbits(16) if initial_sequence is None else initial_sequence
        )
        self._initial_timestamp = (
            secrets.randbits(32)
            if initial_timestamp is None
            else initial_timestamp
        )
        packetize_h264_rtp(
            b"\x65",
            sequence_start=self._sequence,
            timestamp=self._initial_timestamp,
            ssrc=self._ssrc,
            payload_type=self._payload_type,
            mtu=self._mtu,
        )
        self._clock = clock or MonotonicClock()
        self._socket_factory = socket_factory
        self._socket: DatagramSocket | None = None
        self._started = False
        self._closed = False
        self._stream_id: str | None = None
        self._last_frame_sequence: int | None = None
        self._base_source_timestamp_ns: int | None = None
        self._frames_sent = 0
        self._packets_sent = 0
        self._bytes_sent = 0
        self._dropped_stale = 0
        self._dropped_late = 0
        self._errors = 0

    @property
    def metrics(self) -> RtpH264TransportMetrics:
        return RtpH264TransportMetrics(
            frames_sent=self._frames_sent,
            packets_sent=self._packets_sent,
            bytes_sent=self._bytes_sent,
            dropped_stale=self._dropped_stale,
            dropped_late=self._dropped_late,
            errors=self._errors,
        )

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed RTP/H.264 transport")
        if self._started:
            raise LifecycleError("RTP/H.264 transport is already started")
        self._socket = self._socket_factory()
        self._started = True

    def send(self, frame: EncodedFrame) -> None:
        if self._closed:
            raise LifecycleError("RTP/H.264 transport is closed")
        if not self._started or self._socket is None:
            raise LifecycleError("RTP/H.264 transport has not been started")
        if not isinstance(frame, EncodedFrame):
            raise ModelValidationError("frame must be an EncodedFrame")
        if frame.codec is not VideoCodec.H264:
            raise ModelValidationError("RTP/H.264 transport requires H.264 frames")
        if self._stream_id is not None and frame.stream_id != self._stream_id:
            raise ModelValidationError("one RTP transport can send only one stream_id")
        if (
            self._last_frame_sequence is not None
            and not is_newer_sequence(frame.sequence, self._last_frame_sequence)
        ):
            self._dropped_stale += 1
            return
        age_ns = self._clock.now_ns() - frame.encoded_timestamp_ns
        if age_ns > self._max_age_ns:
            self._dropped_late += 1
            return
        base_source = self._base_source_timestamp_ns
        if base_source is None:
            base_source = frame.source_timestamp_ns
        if frame.source_timestamp_ns < base_source:
            self._dropped_late += 1
            return
        timestamp = (
            self._initial_timestamp
            + (frame.source_timestamp_ns - base_source) * RTP_CLOCK_HZ // 1_000_000_000
        ) & 0xFFFFFFFF
        packets = packetize_h264_rtp(
            frame.data,
            sequence_start=self._sequence,
            timestamp=timestamp,
            ssrc=self._ssrc,
            payload_type=self._payload_type,
            mtu=self._mtu,
        )
        try:
            for packet in packets:
                sent = self._socket.sendto(packet, self._target)
                if sent != len(packet):
                    raise OSError(
                        f"partial UDP datagram send: {sent}/{len(packet)} bytes"
                    )
        except OSError:
            self._errors += 1
            raise
        self._stream_id = frame.stream_id
        self._last_frame_sequence = frame.sequence
        self._base_source_timestamp_ns = base_source
        self._sequence = (self._sequence + len(packets)) & 0xFFFF
        self._frames_sent += 1
        self._packets_sent += len(packets)
        self._bytes_sent += sum(len(packet) for packet in packets)

    def close(self) -> None:
        if self._closed:
            return
        udp_socket = self._socket
        self._socket = None
        self._started = False
        self._closed = True
        if udp_socket is not None:
            udp_socket.close()


@dataclass(slots=True)
class _JitterFrame:
    packets: dict[int, ParsedRtpPacket] = field(default_factory=dict)
    start_sequence: int | None = None
    marker_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class RtpH264JitterMetrics:
    packets_received: int
    duplicate_packets: int
    malformed_packets: int
    dropped_late: int
    dropped_jitter: int
    frames_emitted: int


class RtpH264JitterBuffer:
    """Bound a small number of out-of-order RTP frames and reassemble Annex B."""

    def __init__(
        self,
        *,
        payload_type: int = 96,
        max_frames: int = 3,
        max_packets_per_frame: int = 4096,
    ) -> None:
        if not 0 <= payload_type <= 127:
            raise ModelValidationError("payload_type must be within [0, 127]")
        for value, name in (
            (max_frames, "max_frames"),
            (max_packets_per_frame, "max_packets_per_frame"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelValidationError(f"{name} must be an integer >= 1")
        self._payload_type = payload_type
        self._max_frames = max_frames
        self._max_packets = max_packets_per_frame
        self._frames: OrderedDict[int, _JitterFrame] = OrderedDict()
        self._last_emitted_timestamp: int | None = None
        self._packets_received = 0
        self._duplicates = 0
        self._malformed = 0
        self._dropped_late = 0
        self._dropped_jitter = 0
        self._frames_emitted = 0

    @property
    def metrics(self) -> RtpH264JitterMetrics:
        return RtpH264JitterMetrics(
            packets_received=self._packets_received,
            duplicate_packets=self._duplicates,
            malformed_packets=self._malformed,
            dropped_late=self._dropped_late,
            dropped_jitter=self._dropped_jitter,
            frames_emitted=self._frames_emitted,
        )

    def push(self, packet: bytes | bytearray | memoryview) -> bool:
        try:
            parsed = parse_rtp_packet(packet)
            if parsed.payload_type != self._payload_type:
                raise ModelValidationError("unexpected RTP payload type")
            nal_type = parsed.payload[0] & 0x1F
            if not (1 <= nal_type <= 23 or nal_type == H264_FU_A_TYPE):
                raise ModelValidationError("unsupported RTP H.264 payload type")
            if nal_type == H264_FU_A_TYPE and len(parsed.payload) < 3:
                raise ModelValidationError("FU-A payload is truncated")
        except ModelValidationError:
            self._malformed += 1
            return False
        if self._last_emitted_timestamp is not None and not is_newer_sequence(
            parsed.timestamp,
            self._last_emitted_timestamp,
            2**32,
        ):
            self._dropped_late += 1
            return False
        jitter_frame = self._frames.get(parsed.timestamp)
        if jitter_frame is None:
            if len(self._frames) >= self._max_frames:
                self._frames.popitem(last=False)
                self._dropped_jitter += 1
            jitter_frame = _JitterFrame()
            self._frames[parsed.timestamp] = jitter_frame
        if parsed.sequence in jitter_frame.packets:
            self._duplicates += 1
            return False
        if len(jitter_frame.packets) >= self._max_packets:
            self._frames.pop(parsed.timestamp, None)
            self._dropped_jitter += 1
            return False
        nal_type = parsed.payload[0] & 0x1F
        is_start = nal_type != H264_FU_A_TYPE or bool(parsed.payload[1] & 0x80)
        if is_start and jitter_frame.start_sequence is None:
            jitter_frame.start_sequence = parsed.sequence
        if parsed.marker:
            jitter_frame.marker_sequence = parsed.sequence
        jitter_frame.packets[parsed.sequence] = parsed
        self._packets_received += 1
        return True

    def _ordered_packets(
        self,
        jitter_frame: _JitterFrame,
    ) -> tuple[ParsedRtpPacket, ...] | None:
        start = jitter_frame.start_sequence
        marker = jitter_frame.marker_sequence
        if start is None or marker is None:
            return None
        count = ((marker - start) & 0xFFFF) + 1
        if count > self._max_packets:
            return None
        sequences = tuple((start + offset) & 0xFFFF for offset in range(count))
        if any(sequence not in jitter_frame.packets for sequence in sequences):
            return None
        return tuple(jitter_frame.packets[sequence] for sequence in sequences)

    @staticmethod
    def _reassemble(packets: tuple[ParsedRtpPacket, ...]) -> bytes:
        nal_units: list[bytes] = []
        current_fu: bytearray | None = None
        for packet in packets:
            payload = packet.payload
            nal_type = payload[0] & 0x1F
            if nal_type != H264_FU_A_TYPE:
                if current_fu is not None:
                    raise ModelValidationError("single NAL interrupted FU-A")
                nal_units.append(payload)
                continue
            fu_header = payload[1]
            start = bool(fu_header & 0x80)
            end = bool(fu_header & 0x40)
            if start:
                if current_fu is not None:
                    raise ModelValidationError("nested FU-A start")
                nal_header = (payload[0] & 0xE0) | (fu_header & 0x1F)
                current_fu = bytearray((nal_header,))
            if current_fu is None:
                raise ModelValidationError("FU-A continuation without start")
            current_fu.extend(payload[2:])
            if end:
                nal_units.append(bytes(current_fu))
                current_fu = None
        if current_fu is not None:
            raise ModelValidationError("FU-A frame is missing the end fragment")
        return b"".join(b"\x00\x00\x00\x01" + nal for nal in nal_units)

    def pop_ready(self) -> bytes | None:
        if not self._frames:
            return None
        timestamp, jitter_frame = next(iter(self._frames.items()))
        packets = self._ordered_packets(jitter_frame)
        if packets is None:
            return None
        self._frames.pop(timestamp)
        try:
            access_unit = self._reassemble(packets)
        except ModelValidationError:
            self._malformed += 1
            return None
        self._last_emitted_timestamp = timestamp
        self._frames_emitted += 1
        return access_unit


def create_rtp_h264_udp(
    config: VideoStreamingConfig,
    network: NetworkConfig,
) -> RtpH264UdpTransport:
    """Create an unstarted experimental camera-0 RTP transport."""

    if network.vr_ip is None:
        raise ModelValidationError("RTP/H.264 transport requires network.vr_ip")
    return RtpH264UdpTransport(
        network.vr_ip,
        network.video_rtp_port,
        config,
    )
