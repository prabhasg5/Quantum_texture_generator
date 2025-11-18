// Canvas/Moodboard JavaScript
let canvasItems = [];
let selectedItem = null;
let draggedItem = null;
let isDragging = false;
let isResizing = false;
let dragOffset = { x: 0, y: 0 };
let selectedNoteColor = '#ffd4a3';
let currentProjectId = null;
let currentProjectName = 'Untitled Canvas';

// Initialize canvas
document.addEventListener('DOMContentLoaded', () => {
    initializeCanvas();
    loadRecentGenerations();
    setupEventListeners();
    checkURLParameters();
    checkPendingCanvasItem();
});

function checkURLParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    const projectId = urlParams.get('project');
    
    if (projectId) {
        loadProject(projectId);
    } else {
        loadSavedCanvas(); // Fallback to localStorage for backward compatibility
    }
}

function checkPendingCanvasItem() {
    const pending = sessionStorage.getItem('pendingCanvasItem');
    if (pending) {
        try {
            const data = JSON.parse(pending);
            const item = createCanvasItem(data.type, {
                ...data,
                x: 100,
                y: 100
            });
            canvasItems.push(item);
            sessionStorage.removeItem('pendingCanvasItem');
            showNotification('Generation added to canvas!');
        } catch (error) {
            console.error('Error adding pending item:', error);
        }
    }
}

function initializeCanvas() {
    const canvas = document.getElementById('canvas');
    
    // Canvas click to deselect
    canvas.addEventListener('click', (e) => {
        if (e.target === canvas || e.target.classList.contains('grid-background')) {
            clearSelection();
        }
    });
}

function setupEventListeners() {
    // Upload button
    document.getElementById('upload-btn').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });
    
    document.getElementById('file-input').addEventListener('change', handleFileUpload);
    
    // Tool buttons
    document.getElementById('text-btn').addEventListener('click', openTextModal);
    document.getElementById('note-btn').addEventListener('click', openNoteModal);
    document.getElementById('color-picker-btn').addEventListener('click', openColorModal);
    document.getElementById('clear-selection-btn').addEventListener('click', clearSelection);
    document.getElementById('delete-btn').addEventListener('click', deleteSelected);
    document.getElementById('save-canvas-btn').addEventListener('click', saveCanvas);
    document.getElementById('export-btn').addEventListener('click', exportCanvas);
    
    // Sidebar toggle
    document.getElementById('sidebar-toggle').addEventListener('click', toggleSidebar);
    
    // Text modal
    document.getElementById('text-size').addEventListener('input', (e) => {
        document.getElementById('text-size-value').textContent = e.target.value + 'px';
    });
    
    // Color input sync
    document.getElementById('color-input').addEventListener('input', (e) => {
        document.getElementById('color-hex-input').value = e.target.value;
    });
    
    document.getElementById('color-hex-input').addEventListener('input', (e) => {
        if (e.target.value.match(/^#[0-9A-F]{6}$/i)) {
            document.getElementById('color-input').value = e.target.value;
        }
    });
    
    // Note color selection
    document.querySelectorAll('.note-color').forEach(color => {
        color.addEventListener('click', function() {
            document.querySelectorAll('.note-color').forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
            selectedNoteColor = this.dataset.color;
        });
    });
}

function handleFileUpload(e) {
    const files = e.target.files;
    
    Array.from(files).forEach((file, index) => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (event) => {
                addImageToCanvas(event.target.result, 100 + index * 50, 100 + index * 50);
            };
            reader.readAsDataURL(file);
        }
    });
    
    e.target.value = ''; // Reset input
}

function addImageToCanvas(src, x = 100, y = 100, width = 300) {
    const img = new Image();
    img.onload = () => {
        const aspectRatio = img.height / img.width;
        const height = width * aspectRatio;
        
        const item = createCanvasItem('image', {
            src,
            x,
            y,
            width,
            height
        });
        
        canvasItems.push(item);
    };
    img.src = src;
}

function addTextToCanvas() {
    const text = document.getElementById('text-input').value.trim();
    const size = document.getElementById('text-size').value;
    
    if (!text) {
        alert('Please enter some text');
        return;
    }
    
    const item = createCanvasItem('text', {
        text,
        size: parseInt(size),
        x: 200,
        y: 200
    });
    
    canvasItems.push(item);
    closeTextModal();
    document.getElementById('text-input').value = '';
}

function addNoteToCanvas() {
    const text = document.getElementById('note-input').value.trim();
    
    if (!text) {
        alert('Please enter some text for the note');
        return;
    }
    
    const item = createCanvasItem('note', {
        text,
        color: selectedNoteColor,
        x: 250,
        y: 150
    });
    
    canvasItems.push(item);
    closeNoteModal();
    document.getElementById('note-input').value = '';
}

function addColorSwatch() {
    const color = document.getElementById('color-input').value;
    
    const item = createCanvasItem('color', {
        color,
        x: 300,
        y: 200
    });
    
    canvasItems.push(item);
    closeColorModal();
}

function createCanvasItem(type, data) {
    const canvas = document.getElementById('canvas');
    const div = document.createElement('div');
    div.className = `canvas-item ${type}-item`;
    div.dataset.type = type;
    div.dataset.id = Date.now() + Math.random();
    
    // Set position
    div.style.left = data.x + 'px';
    div.style.top = data.y + 'px';
    
    // Create delete button
    const deleteBtn = document.createElement('div');
    deleteBtn.className = 'item-delete';
    deleteBtn.innerHTML = '&times;';
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        removeCanvasItem(div);
    };
    div.appendChild(deleteBtn);
    
    // Type-specific content
    if (type === 'image') {
        div.style.width = data.width + 'px';
        div.style.height = data.height + 'px';
        const img = document.createElement('img');
        img.src = data.src;
        div.appendChild(img);
    } else if (type === 'text') {
        div.textContent = data.text;
        div.style.fontSize = data.size + 'px';
    } else if (type === 'note') {
        div.style.background = data.color;
        div.textContent = data.text;
    } else if (type === 'color') {
        div.style.background = data.color;
        const label = document.createElement('span');
        label.className = 'color-label';
        label.textContent = data.color;
        div.appendChild(label);
    } else if (type === 'generation') {
        div.style.width = '400px';
        
        // Image
        const img = document.createElement('img');
        img.className = 'generation-image';
        img.src = data.image;
        div.appendChild(img);
        
        // Info container
        const info = document.createElement('div');
        info.className = 'gen-info';
        
        // Colors
        if (data.colors && data.colors.length) {
            const colorsDiv = document.createElement('div');
            colorsDiv.className = 'gen-colors';
            data.colors.forEach(color => {
                const colorDiv = document.createElement('div');
                colorDiv.className = 'gen-color';
                colorDiv.style.background = color;
                colorDiv.title = color;
                colorDiv.onclick = () => copyToClipboard(color);
                colorsDiv.appendChild(colorDiv);
            });
            info.appendChild(colorsDiv);
        }
        
        // Concepts
        if (data.concepts && data.concepts.length) {
            const conceptsDiv = document.createElement('div');
            conceptsDiv.className = 'gen-concepts';
            data.concepts.forEach(concept => {
                const tag = document.createElement('span');
                tag.className = 'gen-keyword';
                tag.textContent = concept;
                tag.onclick = () => copyToClipboard(concept);
                conceptsDiv.appendChild(tag);
            });
            info.appendChild(conceptsDiv);
        }
        
        div.appendChild(info);
    }
    
    // Add resize handle for resizable items
    if (['image', 'generation'].includes(type)) {
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'resize-handle';
        resizeHandle.onmousedown = (e) => startResize(e, div);
        div.appendChild(resizeHandle);
    }
    
    // Make draggable
    makeDraggable(div);
    
    // Make selectable
    div.addEventListener('click', (e) => {
        e.stopPropagation();
        selectItem(div);
    });
    
    canvas.appendChild(div);
    return div;
}

function makeDraggable(element) {
    element.onmousedown = (e) => {
        if (e.target.classList.contains('resize-handle') || e.target.classList.contains('item-delete')) {
            return;
        }
        
        e.preventDefault();
        isDragging = true;
        draggedItem = element;
        element.classList.add('dragging');
        
        const rect = element.getBoundingClientRect();
        const canvasRect = document.getElementById('canvas').getBoundingClientRect();
        
        dragOffset.x = e.clientX - rect.left;
        dragOffset.y = e.clientY - rect.top;
        
        document.onmousemove = (e) => {
            if (isDragging && draggedItem) {
                const canvasRect = document.getElementById('canvas').getBoundingClientRect();
                const x = e.clientX - canvasRect.left - dragOffset.x;
                const y = e.clientY - canvasRect.top - dragOffset.y;
                
                draggedItem.style.left = Math.max(0, x) + 'px';
                draggedItem.style.top = Math.max(0, y) + 'px';
            }
        };
        
        document.onmouseup = () => {
            if (draggedItem) {
                draggedItem.classList.remove('dragging');
            }
            isDragging = false;
            draggedItem = null;
            document.onmousemove = null;
            document.onmouseup = null;
        };
    };
}

function startResize(e, element) {
    e.preventDefault();
    e.stopPropagation();
    isResizing = true;
    
    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = element.offsetWidth;
    const startHeight = element.offsetHeight;
    
    const aspectRatio = startHeight / startWidth;
    
    document.onmousemove = (e) => {
        if (isResizing) {
            const newWidth = startWidth + (e.clientX - startX);
            const newHeight = newWidth * aspectRatio;
            
            if (newWidth > 50) {
                element.style.width = newWidth + 'px';
                element.style.height = newHeight + 'px';
            }
        }
    };
    
    document.onmouseup = () => {
        isResizing = false;
        document.onmousemove = null;
        document.onmouseup = null;
    };
}

function selectItem(element) {
    clearSelection();
    selectedItem = element;
    element.classList.add('selected');
}

function clearSelection() {
    if (selectedItem) {
        selectedItem.classList.remove('selected');
        selectedItem = null;
    }
}

function deleteSelected() {
    if (selectedItem) {
        removeCanvasItem(selectedItem);
    }
}

function removeCanvasItem(element) {
    element.remove();
    canvasItems = canvasItems.filter(item => item !== element);
    if (selectedItem === element) {
        selectedItem = null;
    }
}

async function loadRecentGenerations() {
    const container = document.getElementById('recent-generations');
    
    try {
        const response = await fetch('/api/generations/history');
        const data = await response.json();
        
        if (response.ok && data.generations.length > 0) {
            container.innerHTML = '';
            
            data.generations.slice(0, 10).forEach(gen => {
                const card = createGenerationCard(gen);
                container.appendChild(card);
            });
        } else {
            container.innerHTML = '<div class="loading">No generations yet</div>';
        }
    } catch (error) {
        console.error('Error loading generations:', error);
        container.innerHTML = '<div class="loading">Failed to load</div>';
    }
}

function createGenerationCard(gen) {
    const card = document.createElement('div');
    card.className = 'generation-card';
    
    // Show image if available
    if (gen.image_data) {
        const img = document.createElement('img');
        img.src = gen.image_data;
        img.alt = gen.texture_class;
        img.style.cssText = 'width: 100%; height: 100px; object-fit: cover; border-radius: 4px; margin-bottom: 0.5rem;';
        card.appendChild(img);
    }
    
    // Info section
    const info = document.createElement('div');
    info.className = 'card-info';
    info.innerHTML = `
        <strong>${gen.texture_class.replace(/-/g, ' ')}</strong><br>
        <span style="font-size: 0.85rem; color: var(--text-secondary);">
            ${gen.num_samples} samples • ${formatDate(gen.created_at)}
        </span>
    `;
    card.appendChild(info);
    
    const btn = document.createElement('button');
    btn.className = 'add-to-canvas-btn';
    btn.textContent = 'Add to Canvas';
    btn.onclick = () => addGenerationToCanvas(gen);
    card.appendChild(btn);
    
    return card;
}

async function addGenerationToCanvas(gen) {
    // Extract color palette and concepts from generation_metadata
    const metadata = gen.generation_metadata || {};
    const colors = metadata.color_palette || [];
    const concepts = metadata.concept_words || [];
    
    // Create generation item with image, colors, and concepts
    const item = createCanvasItem('generation', {
        image: gen.image_data,
        colors: colors,
        concepts: concepts,
        color: '#b4ff9f',
        x: 150,
        y: 150
    });
    
    canvasItems.push(item);
}

function saveCanvas() {
    // Get canvas data
    const canvasData = {
        items: []
    };
    
    document.querySelectorAll('.canvas-item').forEach(item => {
        const itemData = {
            type: item.dataset.type,
            id: item.dataset.id,
            x: parseInt(item.style.left),
            y: parseInt(item.style.top),
            width: item.style.width ? parseInt(item.style.width) : null,
            height: item.style.height ? parseInt(item.style.height) : null
        };
        
        // Type-specific data
        if (item.dataset.type === 'text') {
            itemData.text = item.textContent;
            itemData.size = parseInt(item.style.fontSize);
        } else if (item.dataset.type === 'note') {
            itemData.text = item.textContent;
            itemData.color = item.style.background;
        } else if (item.dataset.type === 'color') {
            itemData.color = item.style.background;
        } else if (item.dataset.type === 'image') {
            const img = item.querySelector('img');
            if (img) itemData.src = img.src;
        } else if (item.dataset.type === 'generation') {
            const img = item.querySelector('.generation-image');
            if (img) itemData.image = img.src;
            
            const colors = [];
            item.querySelectorAll('.gen-color').forEach(c => {
                colors.push(c.style.backgroundColor);
            });
            itemData.colors = colors;
            
            const concepts = [];
            item.querySelectorAll('.gen-keyword').forEach(k => {
                concepts.push(k.textContent);
            });
            itemData.concepts = concepts;
        }
        
        canvasData.items.push(itemData);
    });
    
    // Prompt for project name if new project
    let projectName = currentProjectName;
    if (!currentProjectId) {
        projectName = prompt('Enter a name for this canvas:', currentProjectName);
        if (!projectName) return; // Cancelled
        currentProjectName = projectName;
    }
    
    // Generate thumbnail using html2canvas
    generateCanvasThumbnail().then(thumbnail => {
        // Save to API
        saveToAPI(projectName, canvasData, thumbnail);
    }).catch(error => {
        console.error('Error generating thumbnail:', error);
        // Save without thumbnail if generation fails
        saveToAPI(projectName, canvasData, null);
    });
}

async function saveToAPI(name, canvasData, thumbnail) {
    try {
        const response = await fetch('/api/projects/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                project_id: currentProjectId,
                name: name,
                canvas_data: canvasData,
                thumbnail: thumbnail
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentProjectId = data.project_id;
            showNotification('Canvas saved successfully!');
            
            // Update URL without reload
            window.history.replaceState({}, '', `/canvas?project=${currentProjectId}`);
        } else {
            alert(data.error || 'Failed to save canvas');
        }
    } catch (error) {
        console.error('Error saving canvas:', error);
        alert('An error occurred while saving');
    }
}

async function generateCanvasThumbnail() {
    // Use html2canvas to capture the canvas as a thumbnail
    try {
        const canvasContainer = document.querySelector('.canvas-container');
        
        // Temporarily hide UI elements
        const sidebar = document.getElementById('sidebar');
        const toolbar = document.querySelector('.toolbar');
        const wasSidebarVisible = sidebar && !sidebar.classList.contains('collapsed');
        
        if (sidebar) sidebar.style.display = 'none';
        if (toolbar) toolbar.style.display = 'none';
        
        // Capture at smaller scale for thumbnail
        const canvas = await html2canvas(canvasContainer, {
            backgroundColor: '#0a0a0f',
            scale: 0.5, // Smaller scale for thumbnail
            logging: false,
            useCORS: true,
            width: 800,
            height: 600
        });
        
        // Restore UI
        if (sidebar) sidebar.style.display = '';
        if (toolbar) toolbar.style.display = '';
        if (wasSidebarVisible && sidebar) {
            sidebar.classList.remove('collapsed');
        }
        
        // Convert to data URL with compression
        return canvas.toDataURL('image/jpeg', 0.7);
    } catch (error) {
        console.error('Error generating thumbnail with html2canvas:', error);
        
        // Fallback: try to use first image or create placeholder
        const firstImage = document.querySelector('.canvas-item img');
        if (firstImage && firstImage.src) {
            return firstImage.src;
        }
        
        // Last resort: create a simple placeholder
        const fallbackCanvas = document.createElement('canvas');
        fallbackCanvas.width = 400;
        fallbackCanvas.height = 300;
        const ctx = fallbackCanvas.getContext('2d');
        
        // Gradient background
        const gradient = ctx.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 245, 255, 0.2)');
        gradient.addColorStop(1, 'rgba(255, 0, 255, 0.2)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 400, 300);
        
        // Add text
        ctx.fillStyle = '#ffffff';
        ctx.font = '24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(currentProjectName || 'Canvas', 200, 150);
        
        return fallbackCanvas.toDataURL('image/png');
    }
}

async function loadProject(projectId) {
    try {
        const response = await fetch(`/api/projects/load/${projectId}`);
        const data = await response.json();
        
        if (response.ok) {
            currentProjectId = data.project.id;
            currentProjectName = data.project.name;
            
            // Clear existing items
            document.querySelectorAll('.canvas-item').forEach(item => item.remove());
            canvasItems = [];
            
            // Load items
            if (data.project.canvas_data && data.project.canvas_data.items) {
                data.project.canvas_data.items.forEach(itemData => {
                    const item = createCanvasItem(itemData.type, itemData);
                    canvasItems.push(item);
                });
            }
            
            showNotification(`Loaded project: ${currentProjectName}`);
        } else {
            alert(data.error || 'Failed to load project');
        }
    } catch (error) {
        console.error('Error loading project:', error);
        alert('An error occurred while loading the project');
    }
}

function loadSavedCanvas() {
    // Fallback to localStorage for backward compatibility
    const saved = localStorage.getItem('canvas_data');
    if (!saved) return;
    
    try {
        const canvasData = JSON.parse(saved);
        
        canvasData.items.forEach(itemData => {
            const item = createCanvasItem(itemData.type, itemData);
            canvasItems.push(item);
        });
    } catch (error) {
        console.error('Error loading canvas:', error);
    }
}

async function exportCanvas() {
    // Export canvas as PNG image using html2canvas
    try {
        const canvasContainer = document.querySelector('.canvas-container');
        
        // Temporarily hide any UI elements we don't want in the export
        const sidebar = document.getElementById('sidebar');
        const toolbar = document.querySelector('.toolbar');
        const wasCollapsed = sidebar.classList.contains('collapsed');
        
        sidebar.style.display = 'none';
        toolbar.style.display = 'none';
        
        // Capture the canvas
        const canvas = await html2canvas(canvasContainer, {
            backgroundColor: '#0a0a0f',
            scale: 2, // Higher quality
            logging: false,
            useCORS: true // Allow cross-origin images
        });
        
        // Restore UI elements
        sidebar.style.display = '';
        toolbar.style.display = '';
        if (wasCollapsed) {
            sidebar.classList.add('collapsed');
        }
        
        // Convert to blob and download
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const projectNameSlug = currentProjectName.toLowerCase().replace(/\s+/g, '-');
            link.setAttribute('href', url);
            link.setAttribute('download', `${projectNameSlug}-${Date.now()}.png`);
            link.click();
            URL.revokeObjectURL(url);
            
            showNotification('Canvas exported as PNG!');
        }, 'image/png');
        
    } catch (error) {
        console.error('Error exporting canvas:', error);
        showNotification('Export failed. Please try again.');
    }
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

function openTextModal() {
    document.getElementById('text-modal').classList.add('active');
}

function closeTextModal() {
    document.getElementById('text-modal').classList.remove('active');
}

function openNoteModal() {
    document.getElementById('note-modal').classList.add('active');
}

function closeNoteModal() {
    document.getElementById('note-modal').classList.remove('active');
}

function openColorModal() {
    document.getElementById('color-modal').classList.add('active');
}

function closeColorModal() {
    document.getElementById('color-modal').classList.remove('active');
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
        top: 140px;
        right: 20px;
        background: linear-gradient(135deg, #00f5ff, #ff00ff);
        color: #0a0a0f;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        z-index: 3000;
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
