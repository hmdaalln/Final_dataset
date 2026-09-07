const byId = id => document.getElementById(id);
const fileInput = byId('image');
let busy = false;
function resetResults() {
  for (const id of ['original', 'result']) { byId(id).hidden = true; byId(id).removeAttribute('src'); }
  byId('originalEmpty').hidden = false;
  byId('resultEmpty').hidden = false;
  byId('download').hidden = true;
  byId('download').removeAttribute('href');
}
fileInput.addEventListener('change', () => {
  resetResults();
  const file = fileInput.files[0];
  byId('filename').textContent = file ? file.name : 'No image selected';
  byId('submit').disabled = !file;
  byId('status').textContent = file ? 'Ready to denoise.' : 'Choose an image to begin.';
  byId('details').textContent ='';
});
byId('uploadForm').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const file = fileInput.files[0];
  if (!file) return;
  resetResults();
  if (file.size > 50 * 1024 * 1024) { byId('status').textContent = 'Please choose an image smaller than 10 MB.'; return; }
  const formData = new FormData(); formData.append('image', file);
  busy = true; fileInput.disabled = true; byId('submit').disabled = true;
  byId('submit').textContent = 'Processing…';
  byId('status').textContent = 'DnCNN is processing your image. Please wait; CPU inference can take a minute or more.';
  try {
    const response = await fetch('/denoise', { method: 'POST', body: formData });
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unexpected response. Check that app.py is running.'); }
    if (!response.ok) throw new Error(data.error || 'Processing failed.');
    byId('original').src = data.original; byId('original').hidden = false;
    byId('result').src = data.result; byId('result').hidden = false;
    byId('originalEmpty').hidden = true; byId('resultEmpty').hidden = true;
    byId('download').href = data.result; byId('download').hidden = false;
    byId('details').textContent = '';
  } catch (error) {
    byId('status').textContent = error instanceof TypeError ? 'Cannot reach the app. Make sure app.py is running and try again.' : error.message;
  } finally {
    busy = false; fileInput.disabled = false; byId('submit').disabled = false; byId('submit').textContent = 'Denoise image ↗';
  }
});
