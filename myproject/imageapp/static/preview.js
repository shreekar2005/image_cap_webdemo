document.addEventListener("DOMContentLoaded", function () {
    const input = document.querySelector('input[type="file"]');
    const previewBox = document.getElementById('imagePreview');
    const previewImg = document.getElementById('preview');
    const uploadArea = document.getElementById('uploadArea');

    // Initially hide the preview box
    previewBox.style.display = 'none';

    // Check for stored image on page load
    const storedImage = sessionStorage.getItem('selectedImage');
    if (storedImage) {
        showPreview(storedImage);
        // Also update the upload area to show we have an image
        uploadArea.innerHTML = `
            <i class="fas fa-check"></i>
            <p>Selected Image</p>
            <span class="file-types">Image ready for captioning</span>
        `;
    }

    input.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            // Check if file is an image
            if (!file.type.match('image.*')) {
                alert('Please select an image file (JPG, PNG, WEBP)');
                return;
            }

            const reader = new FileReader();

            reader.onload = function (e) {
                // Store the image in sessionStorage
                sessionStorage.setItem('selectedImage', e.target.result);
                showPreview(e.target.result);
                
                // Update upload area to show selected file
                uploadArea.innerHTML = `
                    <i class="fas fa-check"></i>
                    <p>${file.name}</p>
                    <span class="file-types">${(file.size / 1024).toFixed(2)} KB</span>
                `;
            };

            reader.onerror = function() {
                uploadArea.innerHTML = `
                    <i class="fas fa-times"></i>
                    <p>Error loading file</p>
                    <span class="file-types">Please try another image</span>
                `;
            };

            reader.readAsDataURL(file);
        }
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--light-gray)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--light-gray)';
        
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            const event = new Event('change');
            input.dispatchEvent(event);
        }
    });
});

function showPreview(imageData) {
    const previewBox = document.getElementById('imagePreview');
    const previewImg = document.getElementById('preview');
    
    previewBox.style.display = "block";
    previewImg.src = imageData;
}

// Clear stored image when form is submitted (optional)
document.querySelector('.upload-form').addEventListener('submit', function() {
    // We don't clear it here because we want it to persist after submission
    // sessionStorage.removeItem('selectedImage');
});