// Custom JavaScript for Document Processing System

document.addEventListener('DOMContentLoaded', function() {
    // File upload area drag and drop functionality
    const fileUploadArea = document.querySelector('.file-upload-area');
    const fileInput = document.getElementById('file');

    if (fileUploadArea && fileInput) {
        fileUploadArea.addEventListener('click', function() {
            fileInput.click();
        });

        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const fileName = this.files[0].name;
                fileUploadArea.querySelector('.file-name').textContent = fileName;
                fileUploadArea.classList.add('has-file');
            } else {
                fileUploadArea.querySelector('.file-name').textContent = 'No file selected';
                fileUploadArea.classList.remove('has-file');
            }
        });

        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            fileUploadArea.classList.add('dragover');
        }

        function unhighlight() {
            fileUploadArea.classList.remove('dragover');
        }

        fileUploadArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length > 0) {
                fileInput.files = files;
                const fileName = files[0].name;
                fileUploadArea.querySelector('.file-name').textContent = fileName;
                fileUploadArea.classList.add('has-file');
            }
        }
    }

    // Processor selection logic
    const allProcessorsCheckbox = document.getElementById('all-processors');
    const processorOptions = document.querySelectorAll('.processor-option');

    if (allProcessorsCheckbox && processorOptions.length > 0) {
        allProcessorsCheckbox.addEventListener('change', function() {
            if (this.checked) {
                processorOptions.forEach(option => {
                    option.style.display = 'none';
                    option.querySelector('input').checked = false;
                });
            } else {
                processorOptions.forEach(option => {
                    option.style.display = 'block';
                });
            }
        });
    }

    // Page navigation in results view
    const pageSelectors = document.querySelectorAll('.page-selector');

    if (pageSelectors.length > 0) {
        pageSelectors.forEach(selector => {
            selector.addEventListener('change', function() {
                const processor = this.dataset.processor;
                const page = this.value;
                const contentDiv = document.getElementById(`${processor}-content`);
                const baseName = document.querySelector('[data-base-name]').dataset.baseName;

                if (!page) {
                    contentDiv.innerHTML = '<p class="text-muted">Select a page to view content</p>';
                    return;
                }

                contentDiv.innerHTML = '<p class="text-muted">Loading content...</p>';

                fetch(`/api/content/${baseName}/${processor}/${page}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.content) {
                            contentDiv.innerHTML = `<pre>${data.content}</pre>`;
                        } else {
                            contentDiv.innerHTML = '<p class="text-muted">No content available</p>';
                        }
                    })
                    .catch(error => {
                        console.error('Error loading content:', error);
                        contentDiv.innerHTML = '<p class="text-danger">Error loading content</p>';
                    });
            });
        });
    }

    // Compare form handling
    const compareForm = document.getElementById('compare-form');

    if (compareForm) {
        const proc1Select = document.getElementById('proc1');
        const proc2Select = document.getElementById('proc2');
        const pageSelect = document.getElementById('page');

        // Update page options when processor 1 changes
        if (proc1Select) {
            proc1Select.addEventListener('change', function() {
                const processor = this.value;
                const baseName = document.querySelector('[data-base-name]').dataset.baseName;

                pageSelect.innerHTML = '<option value="">Loading pages...</option>';

                fetch(`/api/pages/${baseName}/${processor}`)
                    .then(response => response.json())
                    .then(pages => {
                        pageSelect.innerHTML = '';

                        if (pages.length === 0) {
                            const option = document.createElement('option');
                            option.value = '';
                            option.textContent = 'No pages found';
                            pageSelect.appendChild(option);
                        } else {
                            pages.forEach(page => {
                                const option = document.createElement('option');
                                option.value = page;
                                option.textContent = `Page ${page}`;
                                pageSelect.appendChild(option);
                            });
                        }
                    })
                    .catch(error => {
                        console.error('Error loading pages:', error);
                        pageSelect.innerHTML = '<option value="">Error loading pages</option>';
                    });
            });
        }

        // Handle form submission
        compareForm.addEventListener('submit', function(event) {
            event.preventDefault();

            const proc1 = proc1Select.value;
            const proc2 = proc2Select.value;
            const page = pageSelect.value;
            const baseName = document.querySelector('[data-base-name]').dataset.baseName;

            window.location.href = `/compare/${baseName}?proc1=${proc1}&proc2=${proc2}&page=${page}`;
        });
    }
});
