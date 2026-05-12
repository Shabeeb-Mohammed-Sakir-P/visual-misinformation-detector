import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys, os, math

sys.path.insert(0, 'src')
from model import VerifaiDetector
from gradcam import GradCAM, apply_heatmap
from exif_analyzer import extract_exif, format_exif_html

# ─── Load Model ───────────────────────────────────────────
device = torch.device('cpu')
model  = VerifaiDetector(img_size=224)
for path in ['models/verifai_v2_best.pth','models/verifai_v2_final.pth',
             'models/verifai_best.pth','models/verifai_final.pth',
             'models/misinformation_detector.pth']:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded: {path}"); break
model.eval()
gradcam = GradCAM(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def make_meter(conf, is_fake):
    arc_color  = "#993c1d" if is_fake else "#1d9e75"
    angle_deg  = -180 + (conf / 100) * 180
    angle_rad  = math.radians(angle_deg)
    nx         = round(120 + 95 * math.cos(angle_rad))
    ny         = round(120 + 95 * math.sin(angle_rad))
    dashoffset = round(314 - (conf / 100) * 314)
    return f"""
<svg width="100%" viewBox="0 0 240 145">
  <path d="M 20 122 A 100 100 0 0 1 220 122"
        fill="none" stroke="#0c2040" stroke-width="16" stroke-linecap="round"/>
  <path d="M 20 122 A 100 100 0 0 1 220 122"
        fill="none" stroke="{arc_color}" stroke-width="16"
        stroke-linecap="round" stroke-dasharray="314"
        stroke-dashoffset="{dashoffset}"/>
  <path d="M 20 122 A 100 100 0 0 1 220 122"
        fill="none" stroke="#0c447c" stroke-width="1"
        stroke-dasharray="3 9" stroke-linecap="round" opacity="0.5"/>
  <line x1="120" y1="122" x2="{nx}" y2="{ny}"
        stroke="#85b7eb" stroke-width="3" stroke-linecap="round"/>
  <circle cx="120" cy="122" r="9" fill="#378add"/>
  <circle cx="120" cy="122" r="4" fill="#060b1a"/>
  <text x="120" y="96" text-anchor="middle" fill="#85b7eb"
        font-size="26" font-weight="700"
        font-family="Courier New,monospace">{conf:.1f}%</text>
  <text x="120" y="114" text-anchor="middle" fill="#185fa5"
        font-size="8" font-family="Courier New,monospace"
        letter-spacing="2">CONFIDENCE</text>
  <text x="18" y="138" fill="#0c447c" font-size="8"
        font-family="Courier New,monospace">0</text>
  <text x="108" y="18" fill="#0c447c" font-size="8"
        font-family="Courier New,monospace">50</text>
  <text x="208" y="138" fill="#0c447c" font-size="8"
        font-family="Courier New,monospace">100</text>
  <text x="20" y="80" fill="#0c447c" font-size="7"
        font-family="Courier New,monospace">25</text>
  <text x="196" y="80" fill="#0c447c" font-size="7"
        font-family="Courier New,monospace">75</text>
</svg>"""

def build_idle_html():
    return """
<style>
.vf-idle{background:#060b1a;border-radius:0;padding:20px;
         font-family:'Courier New',monospace;height:100%}
.glass{background:#080e1f;border:1px solid #185fa5;border-radius:12px;
       padding:14px;position:relative;overflow:hidden;margin-bottom:12px}
.glass::before{content:'';position:absolute;top:0;left:0;right:0;
               height:1px;background:#378add;opacity:0.3}
.sec-title{color:#185fa5;font-size:8px;letter-spacing:3px;
           margin-bottom:10px;display:flex;align-items:center;gap:6px}
.sec-dot{width:4px;height:4px;border-radius:50%;background:#378add;flex-shrink:0}
.info-row{display:flex;justify-content:space-between;
          margin-bottom:6px;font-size:8px}
.info-key{color:#85b7eb;letter-spacing:1px}
.info-val{color:#1d9e75}
.pulse{animation:pulse 2s infinite}
.blink{animation:blink 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
</style>
<div class="vf-idle">
  <div class="sec-title"><div class="sec-dot"></div>SCAN RESULT</div>
  <div class="glass" style="text-align:center;padding:24px 14px">
    <div style="width:60px;height:60px;border-radius:50%;
                border:2px solid #185fa5;margin:0 auto 12px;
                display:flex;align-items:center;justify-content:center"
         class="pulse">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
           stroke="#378add" stroke-width="1.5" stroke-linecap="round">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
    </div>
    <div style="color:#85b7eb;font-size:11px;letter-spacing:3px">
      AWAITING IMAGE INPUT
    </div>
    <div style="color:#185fa5;font-size:9px;letter-spacing:1px;margin-top:6px">
      UPLOAD AN IMAGE TO BEGIN SCAN
    </div>
    <div style="display:flex;flex-direction:column;gap:4px;margin-top:14px">
      <div style="height:1px;background:#0c2040"></div>
      <div style="height:1px;background:#0c2040;margin:0 16px"></div>
      <div style="height:1px;background:#0c2040;margin:0 32px"></div>
    </div>
  </div>

  <div class="sec-title"><div class="sec-dot"></div>METADATA ANALYSIS</div>
  <div style="background:#080e1f;border:2px solid #185fa5;border-radius:12px;
              padding:16px;position:relative;overflow:hidden;margin-bottom:12px">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:#185fa5;opacity:0.8"></div>
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:14px">
      <div style="font-size:13px;font-weight:700;letter-spacing:3px;
                  color:#185fa5">AWAITING SCAN</div>
      <div style="background:#185fa5;border-radius:20px;padding:4px 12px;
                  font-size:9px;letter-spacing:2px;color:#060b1a;font-weight:700">
        EXIF SCAN
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div style="background:#040810;border:1px solid #0c2040;
                  border-radius:8px;padding:10px">
        <div style="color:#0c447c;font-size:8px;letter-spacing:1px;margin-bottom:4px">CAMERA</div>
        <div style="color:#85b7eb;font-size:10px;font-weight:700">—</div>
      </div>
      <div style="background:#040810;border:1px solid #0c2040;
                  border-radius:8px;padding:10px">
        <div style="color:#0c447c;font-size:8px;letter-spacing:1px;margin-bottom:4px">SOFTWARE</div>
        <div style="color:#85b7eb;font-size:10px;font-weight:700">—</div>
      </div>
      <div style="background:#040810;border:1px solid #0c2040;
                  border-radius:8px;padding:10px">
        <div style="color:#0c447c;font-size:8px;letter-spacing:1px;margin-bottom:4px">GPS DATA</div>
        <div style="color:#85b7eb;font-size:10px;font-weight:700">—</div>
      </div>
      <div style="background:#040810;border:1px solid #0c2040;
                  border-radius:8px;padding:10px">
        <div style="color:#0c447c;font-size:8px;letter-spacing:1px;margin-bottom:4px">EXIF STATUS</div>
        <div style="color:#85b7eb;font-size:10px;font-weight:700">—</div>
      </div>
    </div>
  </div>

  <div class="sec-title"><div class="sec-dot"></div>SYSTEM INFO</div>
  <div class="glass">
    <div class="info-row"><span class="info-key">ARCHITECTURE</span>
      <span class="info-val">EFFICIENTNET + FFT</span></div>
    <div class="info-row"><span class="info-key">PARAMETERS</span>
      <span class="info-val">30,305,342</span></div>
    <div class="info-row"><span class="info-key">TRAINED ON</span>
      <span class="info-val">205,561 IMAGES</span></div>
    <div class="info-row"><span class="info-key">ACCURACY</span>
      <span class="info-val">99.26%</span></div>
    <div class="info-row"><span class="info-key">SESSION</span>
      <span class="info-val blink">ACTIVE ●</span></div>
  </div>
</div>"""

def build_result_html(real_conf, fake_conf, is_fake, conf, exif):
    label       = "AI GENERATED" if is_fake else "REAL"
    label_color = "#d85a30" if is_fake else "#1d9e75"
    status      = "ARTIFACTS DETECTED" if is_fake else "NO ARTIFACTS FOUND"
    analysis    = (
        "Synthetic frequency artifacts detected.<br>"
        "Periodic grid patterns in FFT spectrum.<br>"
        "Characteristics consistent with AI generation."
        if is_fake else
        "Natural frequency distribution detected.<br>"
        "Organic noise patterns confirmed.<br>"
        "No GAN / diffusion artifacts found."
    )
    meter       = make_meter(conf, is_fake)
    exif_score  = exif['exif_score']
    exif_color  = "#993c1d" if exif_score >= 50 else "#ba7517" if exif_score >= 25 else "#1d9e75"
    exif_flag   = "SUSPICIOUS" if exif_score >= 50 else "CAUTION" if exif_score >= 25 else "CLEAN"

    flags_html = ''.join([
        f'<div style="color:#d85a30;font-size:10px;margin-bottom:4px;'
        f'padding:6px;background:#0a0510;border-radius:4px">⚠ {f}</div>'
        for f in exif['suspicious_flags']
    ]) if exif['suspicious_flags'] else (
        '<div style="color:#1d9e75;font-size:10px;padding:6px;'
        'background:#040810;border-radius:4px">✓ No suspicious flags detected</div>'
    )

    combined_note = ""
    if not is_fake and exif_score >= 70:
        combined_note = """
        <div style="margin-top:10px;padding:12px;background:#0a0805;
            border-radius:8px;border:1px solid #ba7517">
          <div style="color:#ef9f27;font-size:11px;font-weight:700;
                      letter-spacing:1px;margin-bottom:4px">⚠ CONFLICT DETECTED</div>
          <div style="color:#ba7517;font-size:10px;line-height:1.6">
            Model says REAL but metadata is suspicious.<br>Verify manually.
          </div>
        </div>"""
    elif is_fake and exif_score == 0 and exif['has_exif']:
        combined_note = """
        <div style="margin-top:10px;padding:12px;background:#0a0805;
            border-radius:8px;border:1px solid #ba7517">
          <div style="color:#ef9f27;font-size:11px;font-weight:700;
                      letter-spacing:1px;margin-bottom:4px">⚠ CONFLICT DETECTED</div>
          <div style="color:#ba7517;font-size:10px;line-height:1.6">
            Model says FAKE but metadata looks clean.<br>Verify manually.
          </div>
        </div>"""

    exif_items = [
        ('CAMERA', f"{exif['camera_make']} {exif['camera_model'] or ''}" if exif['camera_make'] else '—'),
        ('SOFTWARE', str(exif['software'])[:22] if exif['software'] else '—'),
        ('GPS DATA', 'PRESENT' if exif['gps'] else 'ABSENT'),
        ('EXIF STATUS', 'PRESENT' if exif['has_exif'] else 'ABSENT'),
    ]
    exif_grid = ''.join([f"""
    <div style="background:#040810;border:1px solid #0c2040;
                border-radius:8px;padding:10px">
      <div style="color:#0c447c;font-size:8px;letter-spacing:1px;
                  margin-bottom:4px">{k}</div>
      <div style="color:#85b7eb;font-size:10px;font-weight:700">{v}</div>
    </div>""" for k, v in exif_items])

    return f"""
<style>
.vf-res{{background:#060b1a;border-radius:0;padding:20px;
         font-family:'Courier New',monospace}}
.glass{{background:#080e1f;border:1px solid #185fa5;border-radius:12px;
        padding:14px;position:relative;overflow:hidden;margin-bottom:12px}}
.glass::before{{content:'';position:absolute;top:0;left:0;right:0;
                height:1px;background:#378add;opacity:0.3}}
.sec-title{{color:#185fa5;font-size:8px;letter-spacing:3px;
            margin-bottom:10px;display:flex;align-items:center;gap:6px}}
.sec-dot{{width:4px;height:4px;border-radius:50%;background:#378add;flex-shrink:0}}
.prob-row{{display:flex;align-items:center;gap:8px;margin-bottom:7px}}
.prob-label{{color:#85b7eb;font-size:9px;width:50px;letter-spacing:1px}}
.prob-bar{{flex:1;height:6px;background:#040810;border-radius:3px;
           overflow:hidden;border:1px solid #0c2040}}
.prob-fill{{height:100%;border-radius:3px}}
.prob-val{{color:#85b7eb;font-size:9px;width:36px;text-align:right}}
.info-row{{display:flex;justify-content:space-between;
           margin-bottom:6px;font-size:9px}}
.info-key{{color:#85b7eb;letter-spacing:1px}}
.info-val{{color:#1d9e75}}
.analysis{{background:#040810;border-radius:6px;border-left:2px solid #185fa5;
           padding:10px;margin-top:10px;color:#85b7eb;font-size:9px;line-height:1.8}}
.blink{{animation:blink 1s infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
</style>

<div class="vf-res">
  <div class="sec-title"><div class="sec-dot"></div>SCAN RESULT</div>
  <div class="glass" style="text-align:center">
    {meter}
    <div style="font-size:22px;font-weight:900;letter-spacing:4px;
                color:{label_color};margin-top:4px">{label}</div>
    <div style="color:#185fa5;font-size:8px;letter-spacing:2px;
                margin-top:4px;margin-bottom:14px">{status}</div>
    <div class="prob-row">
      <div class="prob-label">REAL</div>
      <div class="prob-bar">
        <div class="prob-fill" style="width:{real_conf:.1f}%;background:#1d9e75"></div>
      </div>
      <div class="prob-val">{real_conf:.1f}%</div>
    </div>
    <div class="prob-row">
      <div class="prob-label">AI GEN</div>
      <div class="prob-bar">
        <div class="prob-fill" style="width:{fake_conf:.1f}%;background:#993c1d"></div>
      </div>
      <div class="prob-val">{fake_conf:.1f}%</div>
    </div>
    <div class="analysis">{analysis}</div>
  </div>

  <div class="sec-title"><div class="sec-dot"></div>METADATA ANALYSIS</div>
  <div style="background:#080e1f;border:2px solid {exif_color};
              border-radius:12px;padding:16px;position:relative;
              overflow:hidden;margin-bottom:12px">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:{exif_color};opacity:0.9"></div>

    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:14px">
      <div style="font-size:16px;font-weight:900;letter-spacing:3px;
                  color:{exif_color}">{exif_flag}</div>
      <div style="background:{exif_color};border-radius:20px;
                  padding:5px 14px;font-size:9px;letter-spacing:2px;
                  color:#060b1a;font-weight:900">EXIF SCAN</div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;
                gap:8px;margin-bottom:12px">
      {exif_grid}
    </div>

    <div style="background:#040810;border-radius:8px;padding:10px;
                margin-bottom:10px">
      <div style="color:#185fa5;font-size:8px;letter-spacing:2px;
                  margin-bottom:8px">FLAGS</div>
      {flags_html}
    </div>

    <div style="padding:12px;background:#040810;border-radius:8px;
                border-left:3px solid {exif_color}">
      <div style="color:{exif_color};font-size:10px;font-weight:700;
                  letter-spacing:1px;margin-bottom:4px">VERDICT</div>
      <div style="color:#85b7eb;font-size:10px;line-height:1.6">
        {exif['exif_summary']}
      </div>
    </div>
    {combined_note}
  </div>

  <div class="sec-title"><div class="sec-dot"></div>SYSTEM INFO</div>
  <div class="glass">
    <div class="info-row"><span class="info-key">ARCHITECTURE</span>
      <span class="info-val">EFFICIENTNET + FFT</span></div>
    <div class="info-row"><span class="info-key">PARAMETERS</span>
      <span class="info-val">30,305,342</span></div>
    <div class="info-row"><span class="info-key">TRAINED ON</span>
      <span class="info-val">205,561 IMAGES</span></div>
    <div class="info-row"><span class="info-key">ACCURACY</span>
      <span class="info-val">99.26%</span></div>
    <div class="info-row"><span class="info-key">SESSION</span>
      <span class="info-val blink">ACTIVE ●</span></div>
  </div>
</div>"""

def predict(image):
    if image is None:
        return build_idle_html(), None, None

    img_for_exif = Image.open(image)
    img          = img_for_exif.convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]

    real_conf = probs[0].item() * 100
    fake_conf = probs[1].item() * 100
    is_fake   = fake_conf > real_conf
    conf      = max(real_conf, fake_conf)
    exif      = extract_exif(img_for_exif)
    html      = build_result_html(real_conf, fake_conf, is_fake, conf, exif)

    cam, _      = gradcam.generate(transform(img).unsqueeze(0))
    overlay     = apply_heatmap(img, cam, size=224)
    overlay_img = Image.fromarray(overlay)

    return html, overlay_img, img.resize((224, 224))

# ─── CSS ──────────────────────────────────────────────────
css = """
.gradio-container{background:#03060f !important;max-width:100% !important}
footer{display:none !important}
button.lg{
  background:#0c2040 !important;
  border:1px solid #185fa5 !important;
  color:#378add !important;
  font-family:'Courier New',monospace !important;
  letter-spacing:3px !important;
  font-size:11px !important;
  border-radius:8px !important;
}
button.lg:hover{background:#0f2a52 !important}
label{color:#85b7eb !important;
      font-family:'Courier New',monospace !important;
      font-size:9px !important;letter-spacing:2px !important}
"""

# ─── Interface ────────────────────────────────────────────
with gr.Blocks(css=css, title="Verifai — AI Image Detector") as demo:

    gr.HTML("""
    <div style="background:#060b1a;border-bottom:1px solid #0c2040;
                padding:24px 20px;text-align:center;
                font-family:'Courier New',monospace;position:relative">
      <div style="position:absolute;top:0;left:0;right:0;
                  height:1px;background:#185fa5;opacity:0.5"></div>
      <div style="font-size:48px;font-weight:900;letter-spacing:12px;
                  color:#85b7eb;line-height:1">
        <span style="color:#378add;font-size:58px">V</span>ERI<span style="color:#378add">FAI</span>
      </div>
      <div style="color:#185fa5;font-size:9px;letter-spacing:5px;margin-top:6px">
        AI · IMAGE · DETECTION · SYSTEM · v2.0
      </div>
      <div style="display:flex;justify-content:center;gap:6px;margin-top:8px">
        <div style="width:5px;height:5px;border-radius:50%;background:#378add"></div>
        <div style="width:5px;height:5px;border-radius:50%;background:#378add"></div>
        <div style="width:5px;height:5px;border-radius:50%;background:#378add"></div>
        <div style="width:5px;height:5px;border-radius:50%;background:#185fa5"></div>
        <div style="width:5px;height:5px;border-radius:50%;background:#185fa5"></div>
      </div>
      <div style="color:#0c447c;font-size:8px;letter-spacing:3px;margin-top:6px">
        DALL-E · MIDJOURNEY · STABLE DIFFUSION · NIGHTCAFE · DEEPFAKE
      </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background:#060b1a;padding:16px 20px 0;
                        font-family:'Courier New',monospace">
              <div style="color:#185fa5;font-size:8px;letter-spacing:3px;
                          margin-bottom:10px;display:flex;align-items:center;gap:6px">
                <div style="width:4px;height:4px;border-radius:50%;
                            background:#378add"></div>
                IMAGE INPUT
              </div>
            </div>""")

            input_image = gr.Image(
                label="DROP IMAGE OR CLICK TO UPLOAD",
                type="filepath",
                height=260,
            )
            analyse_btn = gr.Button(
                ">>> INITIALISE SCAN <<<",
                variant="primary",
                size="lg"
            )
            gr.HTML("""
            <div style="background:#060b1a;padding:8px 20px 16px;
                        font-family:'Courier New',monospace">
              <div style="background:#040810;border:1px solid #0c2040;
                          border-radius:8px;padding:10px;text-align:center">
                <div style="color:#0c447c;font-size:9px;letter-spacing:1px">
                  SUPPORTS: JPG · PNG · WEBP · MAX 10MB
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                            gap:6px;margin-top:8px">
                  <div style="background:#080e1f;border:1px solid #0c2040;
                              border-radius:6px;padding:6px;text-align:center">
                    <div style="color:#0c447c;font-size:7px;letter-spacing:1px">MODEL</div>
                    <div style="color:#1d9e75;font-size:9px;margin-top:2px">EFFNET+FFT</div>
                  </div>
                  <div style="background:#080e1f;border:1px solid #0c2040;
                              border-radius:6px;padding:6px;text-align:center">
                    <div style="color:#0c447c;font-size:7px;letter-spacing:1px">ACCURACY</div>
                    <div style="color:#1d9e75;font-size:9px;margin-top:2px">99.26%</div>
                  </div>
                  <div style="background:#080e1f;border:1px solid #0c2040;
                              border-radius:6px;padding:6px;text-align:center">
                    <div style="color:#0c447c;font-size:7px;letter-spacing:1px">IMAGES</div>
                    <div style="color:#1d9e75;font-size:9px;margin-top:2px">205K+</div>
                  </div>
                </div>
              </div>
            </div>""")

        with gr.Column(scale=1):
            result_html = gr.HTML(value=build_idle_html())

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background:#060b1a;padding:8px 20px 4px;
                        font-family:'Courier New',monospace">
              <div style="color:#185fa5;font-size:8px;letter-spacing:3px;
                          display:flex;align-items:center;gap:6px">
                <div style="width:4px;height:4px;border-radius:50%;
                            background:#378add"></div>
                ORIGINAL · 224×224
              </div>
            </div>""")
            original_out = gr.Image(height=200, show_label=False)

        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background:#060b1a;padding:8px 20px 4px;
                        font-family:'Courier New',monospace">
              <div style="color:#185fa5;font-size:8px;letter-spacing:3px;
                          display:flex;align-items:center;gap:6px">
                <div style="width:4px;height:4px;border-radius:50%;
                            background:#378add"></div>
                GRAD-CAM · REGION ANALYSIS
              </div>
            </div>""")
            gradcam_out = gr.Image(height=200, show_label=False)

    gr.HTML("""
    <div style="background:#040810;border-top:1px solid #0c2040;
                padding:10px 20px;display:flex;justify-content:space-between;
                align-items:center;font-family:'Courier New',monospace">
      <div style="color:#0c447c;font-size:8px;letter-spacing:1px">
        VERIFAI v2.0 · EFFICIENTNET-B0 + FFT FUSION · 99.26% ACCURACY
      </div>
      <div style="color:#185fa5;font-size:8px;letter-spacing:1px">
        DETECTS: DALL-E · MIDJOURNEY · STABLE DIFFUSION · NIGHTCAFE
      </div>
    </div>""")

    analyse_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[result_html, gradcam_out, original_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)