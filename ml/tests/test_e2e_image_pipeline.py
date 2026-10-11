import urllib.request, json, base64

img_path = r'C:\Users\Rushikesh\.gemini\antigravity-ide\brain\a7c1816d-e024-491e-8d88-4f0433e49c77\.user_uploaded\media_1791024689239.png'
with open(img_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

payload = json.dumps({
    'input_type': 'fft_image',
    'image_base64': b64,
    'source_filename': 'Vibration FFT Spectrum.png',
    'bearing_location': 'Drive End (DE)'
}).encode('utf-8')

req = urllib.request.Request('http://localhost:5000/api/bearing-analysis', data=payload, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode())
d = data['data']
print('Backend Result -> Predicted Class:', d['predicted_class'], '| Confidence:', f"{d['prediction_probability']*100:.1f}%")
print('Probabilities:', {p['name']: f"{p['value']}%" for p in d['probabilities']})
