const form = document.querySelector('#upload');
const source = document.querySelector('#source');
const button = document.querySelector('#run');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const original = document.querySelector('#original');
const cleaned = document.querySelector('#cleaned');
const download = document.querySelector('#download');
let urls = [];

function clearResult() {
  result.hidden = true;
  original.removeAttribute('src');
  cleaned.removeAttribute('src');
  download.removeAttribute('href');
  urls.forEach(url => URL.revokeObjectURL(url));
  urls = [];
}

source.addEventListener('change', () => {
  clearResult();
  status.textContent = '';
  button.disabled = !source.files.length;
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  const file = source.files[0];
  if (!file || source.disabled) return;
  clearResult();
  source.disabled = true;
  button.disabled = true;
  status.textContent = 'Removing stains…';
  try {
    const response = await fetch('/remove-stains', {
      method: 'POST',
      body: file,
      headers: { 'Content-Type': 'application/octet-stream' },
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error);
    }
    const images = await response.formData();
    urls = ['original', 'preview', 'cleaned'].map(name => URL.createObjectURL(images.get(name)));
    original.src = urls[0];
    cleaned.src = urls[1];
    download.href = urls[2];
    download.download = file.name.replace(/\.[^.]+$/, '') + '-cleaned.png';
    result.hidden = false;
    status.textContent = 'Cleaned image ready.';
  } catch (error) {
    status.textContent = error.message || 'Could not clean the image.';
  } finally {
    source.disabled = false;
    button.disabled = !source.files.length;
  }
});
