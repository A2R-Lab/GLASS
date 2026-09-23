// Preserve bookmarks to sections of the old Sphinx homepage.
const oldSections = ['glass-gpu-linear-algebra-simple-subroutines', 'interfaces',
  'measured-defaults', 'quick-start', 'measured-performance'];
if (oldSections.includes(location.hash.slice(1))) {
  location.replace('docs/' + location.search + location.hash);
}
const copyButton = document.querySelector('#copy-citation');
copyButton.addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'Citation copied.';
  } catch {
    status.textContent = 'Copy is unavailable in this browser. Select the citation text below to copy it.';
  }
});
