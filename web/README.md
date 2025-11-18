# Quantum Canvas - Web Interface

Modern landing page and user authentication system for the Quantum Texture Generator.

## Features

- **Immersive 3D Hero**: Holographic particle system with neon aesthetics using Three.js
- **Modern UI**: Glassmorphism, gradient effects, smooth animations
- **User Authentication**: Secure signup/signin with session management
- **Responsive Design**: Mobile-first, works on all devices
- **Database**: SQLite (development) / PostgreSQL (production ready)

## Quick Start

### 1. Install Dependencies

```bash
cd web
pip install -r requirements_web.txt
```

### 2. Run the Server

```bash
python app.py
```

The application will start at `http://localhost:5000`

### 3. Database

The SQLite database is created automatically on first run at `web/instance/quantum_canvas.db`.

For production, set the `DATABASE_URL` environment variable:
```bash
export DATABASE_URL=postgresql://user:password@localhost/quantum_canvas
```

## Project Structure

```
web/
├── app.py                 # Flask backend
├── requirements_web.txt   # Python dependencies
├── templates/
│   └── index.html        # Landing page
├── static/
│   ├── css/
│   │   └── style.css     # Styles (neon, glassmorphism)
│   └── js/
│       └── main.js       # 3D animation, modals, forms
└── instance/
    └── quantum_canvas.db  # SQLite database (auto-created)
```

## API Endpoints

### Authentication
- `POST /api/signup` - Create new account
  ```json
  {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "password": "secure123"
  }
  ```

- `POST /api/signin` - Login
  ```json
  {
    "email": "jane@example.com",
    "password": "secure123"
  }
  ```

- `POST /api/signout` - Logout (clears session)

- `GET /api/user/profile` - Get current user (requires auth)

### Pages
- `GET /` - Landing page
- `GET /dashboard` - User dashboard (requires auth, placeholder)

## Database Models

### User
- `id` (primary key)
- `name`
- `email` (unique, indexed)
- `password_hash`
- `created_at`
- `last_login`
- `is_active`

### Project
- `id` (primary key)
- `user_id` (foreign key → User)
- `name`
- `description`
- `created_at`
- `updated_at`

### Generation
- `id` (primary key)
- `project_id` (foreign key → Project)
- `texture_class`
- `num_samples`
- `file_path`
- `metadata` (JSON)
- `created_at`

## Environment Variables

```bash
# Required for production
SECRET_KEY=your-secret-key-here           # Flask session secret
DATABASE_URL=postgresql://...             # Database connection string

# Optional
FLASK_ENV=development                     # development/production
```

## Production Deployment

1. Set environment variables
2. Use a production WSGI server (gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```
3. Use PostgreSQL instead of SQLite
4. Serve static files via CDN/nginx
5. Enable HTTPS

## Security Notes

- Passwords are hashed using Werkzeug's `generate_password_hash`
- Sessions are signed with `SECRET_KEY`
- CSRF protection recommended for production (Flask-WTF)
- Rate limiting recommended for signup/signin endpoints

## Customization

### Change Neon Colors
Edit CSS variables in `static/css/style.css`:
```css
:root {
    --neon-cyan: #00f5ff;
    --neon-magenta: #ff00ff;
    --neon-yellow: #ffff00;
}
```

### Adjust 3D Particle Count
Edit `static/js/main.js`:
```javascript
const particleCount = 3000; // Lower for better performance
```

## Next Steps

- Build dashboard UI for texture generation
- Integrate with existing QGAN training pipeline
- Add project/generation management endpoints
- Implement file upload/download for textures
- Add WebSocket for real-time generation updates

## Troubleshooting

**Issue**: 3D background not rendering
- Ensure Three.js CDN is loading
- Check browser console for errors
- Try reducing `particleCount` in main.js

**Issue**: Database errors
- Delete `instance/quantum_canvas.db` and restart
- Check file permissions
- Verify SQLite is installed

**Issue**: Session not persisting
- Set `SECRET_KEY` environment variable
- Check cookie settings in browser
- Ensure `session.permanent = True` in signin route
