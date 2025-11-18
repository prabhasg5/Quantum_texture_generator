# Canvas Project Management Features

## Overview
The Quantum Canvas now includes a comprehensive project management system that allows users to save, load, and manage their canvas moodboards persistently in the database.

## New Features

### 1. **Canvas Project Persistence**
- Canvas projects are now saved to the database instead of just localStorage
- Each project includes:
  - Project name
  - Canvas data (all items, positions, sizes, content)
  - Thumbnail preview image
  - Creation and update timestamps
  - Item count

### 2. **Dashboard Project Management**
The dashboard now displays:
- **Canvas Projects Section**: Shows all saved canvas projects with:
  - Thumbnail preview
  - Project name
  - Last updated date
  - Number of items in the canvas
  - "Open Canvas" button to continue editing
  - Delete button (with confirmation)
- **New Canvas Button**: Creates a blank canvas project

### 3. **Canvas Save/Load System**
- **Auto-Save with Project Name**: When saving for the first time, prompts for a project name
- **Update Existing Projects**: Subsequent saves update the existing project
- **Load from URL**: Projects can be opened via URL parameter (`/canvas?project=123`)
- **Backward Compatibility**: Still supports loading from localStorage for old data

### 4. **PNG Export**
- Canvas can now be exported as a high-quality PNG image (2x scale)
- Uses `html2canvas` library to capture the entire canvas
- Hides UI elements (sidebar, toolbar) during export
- Downloads with a clean filename based on project name

### 5. **Generation History with Images**
- Generation history sidebar now displays actual texture images
- Shows image thumbnail, texture class, sample count, and date
- "Add to Canvas" button adds the full generation card (image + colors + concepts)
- Properly extracts metadata (colors and concepts) from database

## Database Schema Updates

### Project Table
- `canvas_data` (JSON): Stores complete canvas state
- `thumbnail` (Text): Base64-encoded preview image
- `project_type` (String): Defaults to 'canvas'

### Generation Table
- `image_data` (Text): Stores full base64 PNG image data

## API Endpoints

### `/api/projects/save` (POST)
Save or update a canvas project
```json
{
  "project_id": 123,  // Optional, for updates
  "name": "My Canvas",
  "canvas_data": { ... },
  "thumbnail": "data:image/png;base64,..."
}
```

### `/api/projects/list` (GET)
Get all canvas projects for the current user
```json
{
  "projects": [
    {
      "id": 123,
      "name": "My Canvas",
      "thumbnail": "data:image/png;base64,..
      "created_at": "2025-11-18T...",
      "updated_at": "2025-11-18T...",
      "item_count": 5
    }
  ]
}
```

### `/api/projects/load/<id>` (GET)
Load a specific canvas project
```json
{
  "project": {
    "id": 123,
    "name": "My Canvas",
    "canvas_data": { ... },
    "created_at": "2025-11-18T...",
    "updated_at": "2025-11-18T..."
  }
}
```

### `/api/projects/delete/<id>` (DELETE)
Delete a canvas project

## User Workflow

1. **Create Textures**
   - Go to Dashboard
   - Select texture class
   - Adjust sample count
   - Click "Generate Textures"
   - View results with color palette and concept words

2. **Build Canvas**
   - Click "Move to Canvas" from dashboard
   - Arrange textures, add notes, images, text, colors
   - Upload custom images
   - Drag and resize all items

3. **Save Project**
   - Click "Save Canvas" button
   - Enter a project name (first time only)
   - Project is saved to database with thumbnail

4. **Manage Projects**
   - Return to Dashboard
   - View all canvas projects in "Canvas Projects" section
   - Click "Open Canvas" to continue editing
   - Delete unwanted projects

5. **Export**
   - Click "Export" button
   - Canvas is captured as PNG image
   - Downloaded to your computer

## Technical Details

### Canvas Item Types
- **image**: Uploaded user images
- **text**: Large heading text
- **note**: Sticky note style text
- **color**: Color swatch with hex code
- **generation**: Full texture card with image, palette, and concepts

### Thumbnail Generation
- Automatically created when saving a project
- Falls back to first image or gradient placeholder
- Used for project cards on dashboard

### Image Storage
- All generation images stored as base64 in database
- Allows immediate display without file system access
- Supports data URLs for easy rendering

## Browser Compatibility
- Modern browsers with ES6+ support
- Requires support for: fetch API, async/await, canvas API
- html2canvas library for PNG export

## Future Enhancements
- Collaborative canvas sharing
- Export to other formats (PDF, SVG)
- Canvas templates
- Advanced search/filter for projects
- Project tags and categories
