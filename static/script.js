// Adds a subtle feedback animation when uploading
document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.querySelector(".upload-form");
  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      const btn = uploadForm.querySelector("button");
      btn.textContent = "Uploading...";
      btn.disabled = true;
    });
  }
});
