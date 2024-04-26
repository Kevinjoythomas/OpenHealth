document.querySelector('.card h2').addEventListener('click', function() {
    var form = document.getElementById('symptomsForm');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
});
