# Images Directory

## Required Photos for About Us Section

Add the following photos to this directory for the About Us section to display properly:

### 1. Fieldwork Photos (2 photos)

**File 1:** `fieldwork1.jpg`
- **Description:** First fieldwork photo from fashion design college visit
- **Recommended Size:** 800x600px minimum (landscape)
- **Format:** JPG or PNG
- **Size Limit:** Under 2MB

**File 2:** `fieldwork2.jpg`
- **Description:** Second fieldwork photo showing team interaction
- **Recommended Size:** 800x600px minimum (landscape)
- **Format:** JPG or PNG
- **Size Limit:** Under 2MB

### 2. Team Member Photos (3 photos)

**File 3:** `prabhas.jpg`
- **Member:** Prabhas Mekala (Roll: 238W1A12G5)
- **Recommended Size:** 500x500px (square, will be displayed as circle)
- **Format:** JPG or PNG
- **Size Limit:** Under 1MB
- **Tip:** Use a clear headshot with plain background for best results

**File 4:** `rishi.jpg`
- **Member:** Rishi Kaushal (Roll: 238W1A12F2)
- **Recommended Size:** 500x500px (square, will be displayed as circle)
- **Format:** JPG or PNG
- **Size Limit:** Under 1MB
- **Tip:** Use a clear headshot with plain background for best results

**File 5:** `ali.jpg`
- **Member:** Shaik Ali Murtuza (Roll: 238W1A12H9)
- **Recommended Size:** 500x500px (square, will be displayed as circle)
- **Format:** JPG or PNG
- **Size Limit:** Under 1MB
- **Tip:** Use a clear headshot with plain background for best results

## How to Add Photos

1. Place all 5 photos in this directory: `/web/static/images/`
2. Make sure filenames match exactly as listed above
3. If you want to use different filenames, update the references in `/web/templates/index.html`

## Current References in index.html

```html
<!-- Fieldwork Photos -->
<img src="{{ url_for('static', filename='images/fieldwork1.jpg') }}" ...>
<img src="{{ url_for('static', filename='images/fieldwork2.jpg') }}" ...>

<!-- Team Photos -->
<img src="{{ url_for('static', filename='images/prabhas.jpg') }}" ...>
<img src="{{ url_for('static', filename='images/rishi.jpg') }}" ...>
<img src="{{ url_for('static', filename='images/ali.jpg') }}" ...>
```

## Photo Optimization Tips

- **Compress images** before uploading to reduce load times
- **Use consistent dimensions** for team photos (all square)
- **Good lighting** makes team photos look professional
- **Plain backgrounds** work best for circular team photos
- **Center faces** in team photos for best circular crop

