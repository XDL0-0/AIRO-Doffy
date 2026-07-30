# AIRO-Doffy Reliable Command Protocol v1

Status: opt-in protocol. Legacy `Start`, `Stop`, and rollback control strings
remain active until composition migration and deployment validation.

## Transport

Commands use a separate WebRTC DataChannel:

```text
label = "commands"
ordered = true
maxRetransmits = null
maxPacketLifeTime = null
```

No command is sent through the realtime state channel. The default ACK deadline
is 1 second and the default receiver dedupe capacity is 1024 command IDs.

## Encoding

Each DataChannel message is one UTF-8 JSON object, no larger than 65,536 bytes.
The canonical encoder sorts keys and omits whitespace. The decoder rejects
unknown, missing, or duplicate fields and unsupported versions/message types.

Command envelope:

```json
{
  "message_type": "command",
  "payload": {
    "clock_domain": "device",
    "command_id": "client-generated-id",
    "kind": "start_recording",
    "origin": "unity",
    "sequence": 7,
    "source_timestamp_ns": 123456789,
    "value": null
  },
  "version": 1
}
```

ACK envelope:

```json
{
  "message_type": "ack",
  "payload": {
    "clock_domain": "monotonic",
    "command_id": "client-generated-id",
    "command_sequence": 7,
    "duplicate": false,
    "message": "recording started",
    "status": "accepted",
    "timestamp_ns": 987654321
  },
  "version": 1
}
```

ACK status is one of `accepted`, `rejected`, or `error`.

## Commands

The explicit enum covers:

- start and stop recording;
- roll back the latest episode;
- recalibrate tactile baseline;
- reset wrench baseline;
- reset the teleoperation reference;
- change teleoperation mode;
- change camera zoom;
- change camera resolution;
- request a safe hold;
- request a controlled stop;
- pause, resume, change video profile, and shutdown.

Only mode/profile/zoom/resolution commands carry a non-empty string `value`.

## Idempotency

The receiver stores the canonical payload and resulting ACK under `command_id`:

- same ID and same payload: do not execute again; replay the cached outcome with
  `duplicate=true`;
- same ID and different payload: do not execute; return `rejected`;
- new ID: dispatch once, cache the outcome before sending the ACK;
- capacity overflow: evict the least-recently-used ID.

Caching before ACK send makes retries safe after ACK loss. Once an ID is evicted,
the receiver cannot recognize a later replay, so clients must use unique IDs and
must not retry beyond the deployment's dedupe retention window.

## Routing

The protocol parser produces `RuntimeCommand`. `CommandRouter` selects an
injected handler by enum and returns a typed runtime event:

- success becomes `COMMAND_ACCEPTED`;
- missing handler or `CommandRejectedError` becomes a warning rejection;
- unexpected handler failure becomes an error rejection.

The router does not import or mutate camera, robot, tactile, recording, dataset,
teleoperation, or runtime implementations.
