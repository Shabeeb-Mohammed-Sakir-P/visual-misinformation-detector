from PIL import Image
from PIL.ExifTags import TAGS
import os

def extract_exif(img_path_or_pil):
    """
    Extract and analyse EXIF metadata from an image.
    Returns a dict with findings and a suspicion score.
    """
    try:
        if isinstance(img_path_or_pil, str):
            img = Image.open(img_path_or_pil)
        else:
            img = img_path_or_pil

        exif_data = img._getexif()
    except Exception:
        exif_data = None

    result = {
        'has_exif'        : False,
        'camera_make'     : None,
        'camera_model'    : None,
        'software'        : None,
        'datetime'        : None,
        'gps'             : False,
        'resolution'      : None,
        'flash'           : None,
        'focal_length'    : None,
        'exposure_time'   : None,
        'iso'             : None,
        'suspicious_flags': [],
        'exif_score'      : 0,  # 0=clean, 100=very suspicious
        'exif_summary'    : ''
    }

    # ── No EXIF at all ────────────────────────────────────
    if not exif_data:
        result['suspicious_flags'].append('No EXIF metadata found')
        result['exif_score'] = 40
        result['exif_summary'] = 'No metadata — AI images typically lack EXIF'
        return result

    result['has_exif'] = True

    # ── Parse tags ────────────────────────────────────────
    tag_map = {TAGS.get(k, k): v for k, v in exif_data.items()}

    result['camera_make']  = tag_map.get('Make')
    result['camera_model'] = tag_map.get('Model')
    result['software']     = tag_map.get('Software', '')
    result['datetime']     = tag_map.get('DateTime')
    result['flash']        = tag_map.get('Flash')
    result['iso']          = tag_map.get('ISOSpeedRatings')

    fl = tag_map.get('FocalLength')
    if fl:
        try:
            result['focal_length'] = f"{float(fl):.1f}mm"
        except:
            result['focal_length'] = str(fl)

    et = tag_map.get('ExposureTime')
    if et:
        try:
            result['exposure_time'] = f"1/{int(1/float(et))}s"
        except:
            result['exposure_time'] = str(et)

    gps = tag_map.get('GPSInfo')
    result['gps'] = bool(gps)

    res_x = tag_map.get('XResolution')
    res_y = tag_map.get('YResolution')
    if res_x and res_y:
        try:
            result['resolution'] = f"{int(float(res_x))} x {int(float(res_y))} dpi"
        except:
            pass

    # ── Suspicion checks ──────────────────────────────────
    score = 0
    flags = []

    # Check for AI software signatures
    software = str(result['software']).lower()
    ai_keywords = ['stable diffusion', 'midjourney', 'dall-e', 'firefly',
                   'adobe generative', 'runway', 'novelai', 'invokeai',
                   'automatic1111', 'comfyui', 'diffusers']
    for kw in ai_keywords:
        if kw in software:
            flags.append(f'AI software detected in metadata: {result["software"]}')
            score += 80
            break

    # No camera make/model but has other EXIF
    if not result['camera_make'] and not result['camera_model']:
        flags.append('No camera make/model in metadata')
        score += 30

    # Software is image editor (not camera)
    editor_keywords = ['photoshop', 'gimp', 'lightroom', 'canva',
                       'snapseed', 'pixlr', 'paint']
    for kw in editor_keywords:
        if kw in software:
            flags.append(f'Image edited with: {result["software"]}')
            score += 20
            break

    # Has camera but no GPS (minor — many people disable GPS)
    if result['camera_make'] and not result['gps']:
        pass  # not suspicious alone

    # Clamp score
    score = min(score, 100)
    result['exif_score']      = score
    result['suspicious_flags'] = flags

    # ── Summary ───────────────────────────────────────────
    if score >= 70:
        result['exif_summary'] = 'HIGH SUSPICION — metadata indicates AI/edited origin'
    elif score >= 30:
        result['exif_summary'] = 'MODERATE — some metadata anomalies detected'
    elif result['camera_make']:
        result['exif_summary'] = f"Clean — captured by {result['camera_make']} {result['camera_model'] or ''}"
    else:
        result['exif_summary'] = 'No camera signature found'

    return result


def format_exif_html(exif):
    """Format EXIF result as HTML for Verifai UI"""

    score     = exif['exif_score']
    color     = '#993c1d' if score >= 50 else '#ba7517' if score >= 25 else '#1d9e75'