document.querySelector('.card h2').addEventListener('click', function() {
    var form = document.getElementById('symptomsForm');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
});
function previewImage() {
    var file = document.getElementById("imageUpload").files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        var preview = document.getElementById("preview");
        var imagePreviewDiv = document.getElementById("imagePreview");
        preview.src = e.target.result;
        imagePreviewDiv.style.display = 'block';
    };
    if (file) {
        reader.readAsDataURL(file);
    }
}