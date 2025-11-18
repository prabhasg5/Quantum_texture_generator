// ===== ANIMATED GRADIENT BACKGROUND =====
function initAnimatedBackground() {
    const container = document.getElementById('canvas-container');
    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    let time = 0;
    
    function resize() {
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    }
    
    function drawWaves() {
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear with gradient base
        const bgGradient = ctx.createLinearGradient(0, 0, width, height);
        bgGradient.addColorStop(0, '#0a0a0f');
        bgGradient.addColorStop(0.5, '#1a1a2e');
        bgGradient.addColorStop(1, '#0a0a0f');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);
        
        // Draw flowing quantum waves
        const waveCount = 5;
        for (let i = 0; i < waveCount; i++) {
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            
            for (let x = 0; x < width; x += 5) {
                const y = height / 2 + 
                    Math.sin(x * 0.005 + time * 0.5 + i * 0.8) * 60 * (1 - i * 0.15) +
                    Math.sin(x * 0.008 + time * 0.3 + i) * 40 * (1 - i * 0.2);
                ctx.lineTo(x, y);
            }
            
            ctx.lineTo(width, height);
            ctx.lineTo(0, height);
            ctx.closePath();
            
            // Gradient colors for each wave
            const gradient = ctx.createLinearGradient(0, 0, width, height);
            const colors = [
                ['rgba(0, 245, 255, 0.15)', 'rgba(0, 245, 255, 0.05)'],
                ['rgba(255, 0, 255, 0.12)', 'rgba(255, 0, 255, 0.04)'],
                ['rgba(0, 255, 200, 0.1)', 'rgba(0, 255, 200, 0.03)'],
                ['rgba(255, 100, 200, 0.08)', 'rgba(255, 100, 200, 0.02)'],
                ['rgba(100, 200, 255, 0.06)', 'rgba(100, 200, 255, 0.01)']
            ];
            
            gradient.addColorStop(0, colors[i][0]);
            gradient.addColorStop(1, colors[i][1]);
            ctx.fillStyle = gradient;
            ctx.fill();
        }
        
        // Draw floating quantum particles
        const particleCount = 30;
        for (let i = 0; i < particleCount; i++) {
            const x = (width * ((i * 37) % 100) / 100 + time * 20 * (1 + i % 3)) % width;
            const y = (height * ((i * 73) % 100) / 100 + Math.sin(time * 0.5 + i) * 50) % height;
            const size = 2 + (i % 3);
            
            const particleGradient = ctx.createRadialGradient(x, y, 0, x, y, size * 3);
            particleGradient.addColorStop(0, i % 2 === 0 ? 'rgba(0, 245, 255, 0.8)' : 'rgba(255, 0, 255, 0.8)');
            particleGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            ctx.fillStyle = particleGradient;
            ctx.beginPath();
            ctx.arc(x, y, size * 3, 0, Math.PI * 2);
            ctx.fill();
        }
        
        time += 0.01;
        requestAnimationFrame(drawWaves);
    }
    
    resize();
    window.addEventListener('resize', resize);
    drawWaves();
}

// ===== MODAL MANAGEMENT =====
const signupModal = document.getElementById('signupModal');
const signinModal = document.getElementById('signinModal');
const signupForm = document.getElementById('signupForm');
const signinForm = document.getElementById('signinForm');

// Open signup modal
document.getElementById('getStarted').addEventListener('click', () => {
    signupModal.classList.add('active');
});

document.getElementById('ctaSignup').addEventListener('click', () => {
    signupModal.classList.add('active');
});

// Open signin modal
document.getElementById('openSignIn').addEventListener('click', () => {
    signinModal.classList.add('active');
});

// Close modals
document.getElementById('closeSignup').addEventListener('click', () => {
    signupModal.classList.remove('active');
});

document.getElementById('closeSignin').addEventListener('click', () => {
    signinModal.classList.remove('active');
});

// Switch between modals
document.getElementById('switchToSignIn').addEventListener('click', (e) => {
    e.preventDefault();
    signupModal.classList.remove('active');
    signinModal.classList.add('active');
});

document.getElementById('switchToSignUp').addEventListener('click', (e) => {
    e.preventDefault();
    signinModal.classList.remove('active');
    signupModal.classList.add('active');
});

// Close modal on outside click
window.addEventListener('click', (e) => {
    if (e.target === signupModal) {
        signupModal.classList.remove('active');
    }
    if (e.target === signinModal) {
        signinModal.classList.remove('active');
    }
});

// ===== FORM HANDLING =====
signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('signup-name').value,
        email: document.getElementById('signup-email').value,
        password: document.getElementById('signup-password').value,
        confirm_password: document.getElementById('signup-confirm').value
    };
    
    // Password validation
    if (formData.password !== formData.confirm_password) {
        alert('Passwords do not match!');
        return;
    }
    
    if (formData.password.length < 8) {
        alert('Password must be at least 8 characters long!');
        return;
    }
    
    try {
        const response = await fetch('/api/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Auto sign-in and redirect to dashboard
            sessionStorage.setItem('isSignedIn', 'true');
            window.location.href = '/dashboard';
        } else {
            alert(data.error || 'Sign up failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    }
});

signinForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        email: document.getElementById('signin-email').value,
        password: document.getElementById('signin-password').value
    };
    
    try {
        const response = await fetch('/api/signin', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Set session flag and redirect to dashboard
            sessionStorage.setItem('isSignedIn', 'true');
            window.location.href = '/dashboard';
        } else {
            alert(data.error || 'Sign in failed. Please check your credentials.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    }
});

// ===== SMOOTH SCROLLING =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ===== SCROLL ANIMATIONS =====
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.feature-card, .step').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// ===== GLITCH EFFECT =====
const glitchText = document.querySelector('.glitch');
if (glitchText) {
    setInterval(() => {
        const shouldGlitch = Math.random() > 0.95;
        if (shouldGlitch) {
            glitchText.style.textShadow = `
                ${Math.random() * 10 - 5}px ${Math.random() * 10 - 5}px 0 #00f5ff,
                ${Math.random() * 10 - 5}px ${Math.random() * 10 - 5}px 0 #ff00ff
            `;
            setTimeout(() => {
                glitchText.style.textShadow = '0 0 10px rgba(255, 255, 255, 0.8)';
            }, 50);
        }
    }, 100);
}

// ===== CURSOR TRAIL (OPTIONAL ENHANCEMENT) =====
const createCursorTrail = () => {
    const canvas = document.createElement('canvas');
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '9999';
    document.body.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const particles = [];
    
    document.addEventListener('mousemove', (e) => {
        particles.push({
            x: e.clientX,
            y: e.clientY,
            size: Math.random() * 3 + 1,
            speedX: (Math.random() - 0.5) * 2,
            speedY: (Math.random() - 0.5) * 2,
            color: `hsl(${Math.random() * 60 + 180}, 100%, 50%)`,
            life: 1
        });
    });
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.life;
            ctx.fill();
            
            p.x += p.speedX;
            p.y += p.speedY;
            p.life -= 0.02;
            
            if (p.life <= 0) {
                particles.splice(i, 1);
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
    
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
};

// Uncomment to enable cursor trail
// createCursorTrail();

// ===== NAVIGATION STATE MANAGEMENT =====
function updateNavigation() {
    const signInBtn = document.getElementById('openSignIn');
    const isSignedIn = sessionStorage.getItem('isSignedIn') === 'true';
    
    if (isSignedIn) {
        signInBtn.textContent = 'Sign Out';
        signInBtn.onclick = async (e) => {
            e.preventDefault();
            try {
                const response = await fetch('/api/signout', { method: 'POST' });
                if (response.ok) {
                    sessionStorage.removeItem('isSignedIn');
                    window.location.href = '/';
                }
            } catch (error) {
                console.error('Signout error:', error);
            }
        };
    } else {
        signInBtn.textContent = 'Sign In';
        signInBtn.onclick = null; // Let the modal handler take over
    }
}

// ===== INIT ON LOAD =====
window.addEventListener('DOMContentLoaded', () => {
    initAnimatedBackground();
    updateNavigation();
});
