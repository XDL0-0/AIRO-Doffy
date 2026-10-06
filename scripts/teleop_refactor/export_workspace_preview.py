"""Export scale-accurate design previews; these are not Unity runtime screenshots."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[2] / 'docs/teleop_refactor'
BG, SURFACE, RAISED, TEXT, MUTED, ACCENT = '#111923', '#1b2633', '#263546', '#f1f4f1', '#adbac7', '#9ae3c6'
parts=[]
def rect(x,y,w,h,fill= SURFACE,r=16):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"/>')
def text(value,x,y,size=23,color=TEXT,bold=False):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(value)}</text>')
def button(value,x,y,w,primary=False,h=58):
    rect(x,y,w,h,ACCENT if primary else RAISED)
    parts.append(f'<text x="{x+w/2}" y="{y+h/2+8}" text-anchor="middle" font-size="23" font-weight="700" fill="{BG if primary else TEXT}">{escape(value)}</text>')
def card(title,y,h):
    rect(250,174+y,842,h);text(title,272,174+y+40,24,bold=True)
def build(page):
    parts.clear()
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850"><g font-family="DejaVu Sans, sans-serif">')
    rect(0,0,1200,850,'#090f16',0)
    parts.append('<g transform="translate(40,30)">')
    rect(0,0,1120,760,BG)
    text('DOFFY',30,58,34,bold=True);text('TELEOPERATION / WORKSPACE',250,50,20,MUTED)
    text('Ready to connect',250,85,22,ACCENT);text('CONTROLLERS',810,55,20,MUTED)
    text('FEEDBACK · WAITING',810,88,18,MUTED)
    text(page,252,146,33,bold=True)
    for i,name in enumerate(['Session','Cameras','Alignment','Upper limb','Display','Help']):
        button(name,24,123+i*72,194)
        if name==page:rect(25,136+i*72,3,32,ACCENT,1)
    button('Compact view',24,602,194);button('Bring closer',24,674,194,h=54)
    rect(248,647,844,89)
    button('Start session',264,663,256,True);button('Start recording',538,663,256);button('Recalibrate',812,663,264)
    if page=='Session':
        card('Connect to your workstation',0,176)
        text('Keep Quest and PC on the same network.',272,250,22,MUTED)
        rect(272,271,588,64,BG);text('10.10.131.72',290,312,25)
        button('Apply',878,274,190)
        card('Input & reference',196,142)
        button('Controllers',272,435,247,True);button('Hand tracking',537,435,247);button('Mirror mode',802,435,266)
        text('Hold the grip to control the robot.',256,556,23,MUTED)
        text('Release the grip to pause.',256,587,23,MUTED)
    else:
        card('Video transport',0,144)
        text('Floating views remain in your space while you work.',272,249,21,MUTED)
        button('Use WebRTC',272,259,246,h=50)
        for i,name in enumerate(['1 view','2 views','3 views']):button(name,536+i*176,259,158,primary=i==0,h=50)
        card('Camera windows',160,142)
        button('Add UDP view',272,399,247);button('Close UDP views',537,399,247);button('Arrange views',802,399,266)
        card('Zoom all views',318,126)
        for i,label in enumerate(['x1.0','x1.5','x2.0']):button(label,272+i*267,553,248)
    parts.append('</g>')
    text('Design layout preview · Runtime appearance and XR interaction require Unity / Quest validation.',40,826,16,MUTED)
    parts.append('</g></svg>')
    output=ROOT/f'workspace-{page.lower()}.svg';output.write_text(''.join(parts))
    print(output)
for page in ('Session','Cameras'):build(page)
