// Dashboard JavaScript
let currentTextureClass = null;
let generatedImageData = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadTextureClasses();
    loadCanvasProjects();
    loadGenerationHistory();
    setupEventListeners();
});

function setupEventListeners() {
    const textureSelect = document.getElementById('texture-select');
    const numSamples = document.getElementById('num-samples');
    const sampleCount = document.getElementById('sample-count');
    const generateBtn = document.getElementById('generate-btn');

    // Update sample count display
    numSamples.addEventListener('input', (e) => {
        sampleCount.textContent = e.target.value;
    });

    // Enable generate button when texture is selected
    textureSelect.addEventListener('change', (e) => {
        currentTextureClass = e.target.value;
        generateBtn.disabled = !currentTextureClass;
    });

    // Generate button click
    generateBtn.addEventListener('click', generateTextures);
}

async function loadTextureClasses() {
    try {
        const response = await fetch('/api/texture-classes');
        const data = await response.json();

        if (response.ok) {
            const textureSelect = document.getElementById('texture-select');
            const textureList = document.getElementById('texture-classes');

            // Populate dropdown
            textureSelect.innerHTML = '<option value="">Select a texture class...</option>';
            data.classes.forEach(className => {
                const option = document.createElement('option');
                option.value = className;
                option.textContent = className.replace(/-/g, ' ');
                textureSelect.appendChild(option);
            });

            // Populate sidebar list
            textureList.innerHTML = '';
            data.classes.forEach(className => {
                const item = document.createElement('div');
                item.className = 'texture-item';
                item.textContent = className.replace(/-/g, ' ');
                item.onclick = () => selectTextureFromSidebar(className, item);
                textureList.appendChild(item);
            });
        } else {
            console.error('Failed to load texture classes:', data.error);
        }
    } catch (error) {
        console.error('Error loading texture classes:', error);
    }
}

function selectTextureFromSidebar(className, element) {
    // Update sidebar selection
    document.querySelectorAll('.texture-item').forEach(item => {
        item.classList.remove('active');
    });
    element.classList.add('active');

    // Update dropdown
    const textureSelect = document.getElementById('texture-select');
    textureSelect.value = className;
    currentTextureClass = className;

    // Enable generate button
    document.getElementById('generate-btn').disabled = false;
}

async function generateTextures() {
    const numSamples = parseInt(document.getElementById('num-samples').value);
    const generateBtn = document.getElementById('generate-btn');
    const btnText = generateBtn.querySelector('.btn-text');
    const btnLoading = generateBtn.querySelector('.btn-loading');

    // Show loading state
    generateBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-block';

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texture_class: currentTextureClass,
                num_samples: numSamples
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
            loadGenerationHistory(); // Refresh history
        } else {
            alert(data.error || 'Generation failed. Please try again.');
        }
    } catch (error) {
        console.error('Error generating textures:', error);
        alert('An error occurred. Please try again.');
    } finally {
        // Reset button state
        generateBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    const generatedImage = document.getElementById('generated-image');
    const colorPalette = document.getElementById('color-palette');
    const conceptWords = document.getElementById('concept-words');

    // Store full generation data for download and canvas transfer
    generatedImageData = data;

    // Show results section
    resultsSection.style.display = 'block';

    // Display generated image
    generatedImage.src = data.image;

    // Display color palette
    colorPalette.innerHTML = '';
    data.color_palette.forEach(color => {
        const swatch = document.createElement('div');
        swatch.className = 'color-swatch';
        swatch.style.backgroundColor = color;
        swatch.title = color;
        swatch.onclick = () => copyToClipboard(color);
        
        const hex = document.createElement('span');
        hex.className = 'color-hex';
        hex.textContent = color;
        swatch.appendChild(hex);
        
        colorPalette.appendChild(swatch);
    });

    // Display concept words
    conceptWords.innerHTML = '';
    data.concept_words.forEach(concept => {
        const tag = document.createElement('span');
        tag.className = 'concept-tag';
        tag.textContent = concept;
        tag.onclick = () => copyToClipboard(concept);
        conceptWords.appendChild(tag);
    });

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function loadCanvasProjects() {
    const projectsGrid = document.getElementById('projects-grid');

    try {
        const response = await fetch('/api/projects/list');
        const data = await response.json();

        if (response.ok) {
            if (data.projects.length === 0) {
                projectsGrid.innerHTML = '<div class="loading">No canvas projects yet. Click "New Canvas" to start!</div>';
                return;
            }

            projectsGrid.innerHTML = '';
            data.projects.forEach(project => {
                const card = document.createElement('div');
                card.className = 'project-card';
                
                // Thumbnail
                const thumbnail = document.createElement('div');
                thumbnail.className = 'project-thumbnail';
                if (project.thumbnail) {
                    const img = document.createElement('img');
                    img.src = project.thumbnail;
                    img.alt = project.name;
                    thumbnail.appendChild(img);
                } else {
                    thumbnail.textContent = 'No preview';
                }
                
                // Project info
                const info = document.createElement('div');
                info.className = 'project-info';
                
                const name = document.createElement('div');
                name.className = 'project-name';
                name.textContent = project.name;
                
                const meta = document.createElement('div');
                meta.className = 'project-meta';
                
                const date = document.createElement('span');
                date.className = 'project-date';
                date.textContent = formatDate(project.updated_at);
                
                const items = document.createElement('span');
                items.className = 'project-items';
                items.textContent = `${project.item_count} items`;
                
                meta.appendChild(date);
                meta.appendChild(items);
                
                // Actions
                const actions = document.createElement('div');
                actions.className = 'project-actions';
                
                const openBtn = document.createElement('button');
                openBtn.className = 'btn-open-project';
                openBtn.textContent = 'Open Canvas';
                openBtn.onclick = (e) => {
                    e.stopPropagation();
                    openProject(project.id);
                };
                
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn-delete-project';
                deleteBtn.textContent = '🗑️';
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    deleteProject(project.id, project.name);
                };
                
                actions.appendChild(openBtn);
                actions.appendChild(deleteBtn);
                
                info.appendChild(name);
                info.appendChild(meta);
                info.appendChild(actions);
                
                card.appendChild(thumbnail);
                card.appendChild(info);
                
                // Click card to open
                card.onclick = () => openProject(project.id);
                
                projectsGrid.appendChild(card);
            });
        }
    } catch (error) {
        console.error('Error loading projects:', error);
        projectsGrid.innerHTML = '<div class="loading">Failed to load projects</div>';
    }
}

function createNewCanvas() {
    window.location.href = '/canvas';
}

function openProject(projectId) {
    window.location.href = `/canvas?project=${projectId}`;
}

async function deleteProject(projectId, projectName) {
    if (!confirm(`Delete project "${projectName}"? This cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/projects/delete/${projectId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadCanvasProjects(); // Refresh project list
            showNotification(`Project "${projectName}" deleted`);
        } else {
            const data = await response.json();
            alert(data.error || 'Failed to delete project');
        }
    } catch (error) {
        console.error('Error deleting project:', error);
        alert('An error occurred while deleting the project');
    }
}

async function loadGenerationHistory() {
    const historyList = document.getElementById('history-list');

    try {
        const response = await fetch('/api/generations/history');
        const data = await response.json();

        if (response.ok) {
            if (data.generations.length === 0) {
                historyList.innerHTML = '<div class="loading">No generations yet. Create your first texture!</div>';
                return;
            }

            historyList.innerHTML = '';
            data.generations.forEach(gen => {
                const item = document.createElement('div');
                item.className = 'history-item';
                
                const textureClass = document.createElement('div');
                textureClass.className = 'texture-class';
                textureClass.textContent = gen.texture_class.replace(/-/g, ' ');
                
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.textContent = formatDate(gen.created_at);
                
                const samples = document.createElement('div');
                samples.className = 'timestamp';
                samples.textContent = `${gen.num_samples} samples`;
                
                item.appendChild(textureClass);
                item.appendChild(timestamp);
                item.appendChild(samples);
                
                historyList.appendChild(item);
            });
        }
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = '<div class="loading">Failed to load history</div>';
    }
}

function downloadImage() {
    if (!generatedImageData) return;

    const link = document.createElement('a');
    link.href = generatedImageData.image;
    link.download = `quantum-texture-${currentTextureClass}-${Date.now()}.png`;
    link.click();
}

function moveToCanvas() {
    if (!generatedImageData) {
        alert('No generation to move. Generate textures first!');
        return;
    }

    // Save generation data to sessionStorage for canvas page
    sessionStorage.setItem('pendingCanvasItem', JSON.stringify({
        type: 'generation',
        image: generatedImageData.image,
        colors: generatedImageData.color_palette,
        concepts: generatedImageData.concept_words,
        textureClass: currentTextureClass
    }));

    // Redirect to canvas
    window.location.href = '/canvas';
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification(`Copied: ${text}`);
    });
}

function showNotification(message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #00f5ff, #ff00ff);
        color: #0a0a0f;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 2000);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
}

async function signOut() {
    try {
        const response = await fetch('/api/signout', { method: 'POST' });
        if (response.ok) {
            sessionStorage.removeItem('isSignedIn');
            window.location.href = '/';
        }
    } catch (error) {
        console.error('Signout error:', error);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
