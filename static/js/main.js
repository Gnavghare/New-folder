// Main JavaScript for Fitness Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            } else {
                // Add loading state to submit button
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
                    submitBtn.disabled = true;
                }
                
                // Add loading state to form container
                const formContainer = form.closest('.card');
                if (formContainer) {
                    formContainer.classList.add('loading');
                }
            }
            
            form.classList.add('was-validated');
        }, false);
    });

    // Image upload preview
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const previewContainer = document.querySelector('.image-preview-container');
            const previewImage = document.getElementById('preview-image');
            
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('d-none');
                }
                
                reader.readAsDataURL(this.files[0]);
            } else {
                previewContainer.classList.add('d-none');
            }
        });
    }

    // Dietary restrictions checkbox logic
    const veganCheckbox = document.getElementById('vegan');
    const vegetarianCheckbox = document.getElementById('vegetarian');
    
    if (veganCheckbox && vegetarianCheckbox) {
        veganCheckbox.addEventListener('change', function() {
            if (this.checked) {
                vegetarianCheckbox.checked = true;
                vegetarianCheckbox.disabled = true;
            } else {
                vegetarianCheckbox.disabled = false;
            }
        });
    }

    // Print functionality
    const printButton = document.querySelector('.btn-print');
    if (printButton) {
        printButton.addEventListener('click', function(e) {
            e.preventDefault();
            window.print();
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Accordion state persistence
    const accordions = document.querySelectorAll('.accordion');
    
    accordions.forEach(accordion => {
        const accordionId = accordion.id;
        
        // Check if we have saved state
        const savedState = localStorage.getItem('accordion_' + accordionId);
        
        if (savedState) {
            const openItemId = savedState;
            const openItem = document.getElementById(openItemId);
            
            if (openItem) {
                const bsCollapse = new bootstrap.Collapse(openItem, {
                    toggle: false
                });
                bsCollapse.show();
            }
        }
        
        // Save state when accordion items are clicked
        const accordionButtons = accordion.querySelectorAll('.accordion-button');
        
        accordionButtons.forEach(button => {
            button.addEventListener('click', function() {
                const target = this.getAttribute('data-bs-target').substring(1);
                localStorage.setItem('accordion_' + accordionId, target);
            });
        });
    });
}); 