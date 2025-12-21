"""
Modern Animations Module for Dashboard
======================================

Advanced animations and visual effects using Streamlit components
and modern CSS/JavaScript techniques.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
from typing import Optional, List, Dict, Any
import time


class ModernAnimations:
    """
    Collection of modern animation utilities for the dashboard.
    """
    
    @staticmethod
    def animated_counter(
        value: float,
        label: str,
        prefix: str = "",
        suffix: str = "",
        duration: int = 2000,
        decimals: int = 2,
        color: str = "#4F46E5"
    ):
        """
        Display an animated counter that counts up to the value.
        
        Args:
            value: Target value to animate to
            label: Label below the counter
            prefix: Prefix before the number (e.g., "$")
            suffix: Suffix after the number (e.g., "%")
            duration: Animation duration in milliseconds
            decimals: Number of decimal places
            color: Color of the number
        """
        html = f"""
        <div class="animated-counter-container">
            <div class="counter-value" id="counter-{label.replace(' ', '-')}" 
                 style="color: {color}; font-size: 2.5rem; font-weight: 700;">
                {prefix}0{suffix}
            </div>
            <div class="counter-label" style="color: #64748B; font-size: 0.875rem; margin-top: 5px;">
                {label}
            </div>
        </div>
        <script>
            (function() {{
                const counter = document.getElementById('counter-{label.replace(' ', '-')}');
                const target = {value};
                const duration = {duration};
                const decimals = {decimals};
                const prefix = '{prefix}';
                const suffix = '{suffix}';
                
                let start = 0;
                const startTime = performance.now();
                
                function easeOutQuart(x) {{
                    return 1 - Math.pow(1 - x, 4);
                }}
                
                function animate(currentTime) {{
                    const elapsed = currentTime - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const easedProgress = easeOutQuart(progress);
                    const current = easedProgress * target;
                    
                    counter.textContent = prefix + current.toFixed(decimals) + suffix;
                    
                    if (progress < 1) {{
                        requestAnimationFrame(animate);
                    }}
                }}
                
                requestAnimationFrame(animate);
            }})();
        </script>
        <style>
            .animated-counter-container {{
                text-align: center;
                padding: 20px;
            }}
        </style>
        """
        components.html(html, height=120)
    
    @staticmethod
    def confetti_effect(colors: List[str] = None, duration: int = 3000):
        """
        Display a confetti animation effect for celebrations.
        
        Args:
            colors: List of hex colors for confetti
            duration: Duration in milliseconds
        """
        if colors is None:
            colors = ["#4F46E5", "#7C3AED", "#EC4899", "#10B981", "#F59E0B"]
        
        colors_js = json.dumps(colors)
        
        html = f"""
        <canvas id="confetti-canvas" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9999;"></canvas>
        <script>
        (function() {{
            const canvas = document.getElementById('confetti-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            
            const colors = {colors_js};
            const confetti = [];
            const confettiCount = 150;
            
            for (let i = 0; i < confettiCount; i++) {{
                confetti.push({{
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height - canvas.height,
                    r: Math.random() * 6 + 4,
                    color: colors[Math.floor(Math.random() * colors.length)],
                    tilt: Math.random() * 10 - 5,
                    tiltAngle: Math.random() * Math.PI * 2,
                    tiltAngleIncrement: Math.random() * 0.1 + 0.05,
                    speed: Math.random() * 3 + 2
                }});
            }}
            
            let frame = 0;
            const maxFrames = {duration} / 16;
            
            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                confetti.forEach((c, i) => {{
                    ctx.beginPath();
                    ctx.lineWidth = c.r / 2;
                    ctx.strokeStyle = c.color;
                    ctx.moveTo(c.x + c.tilt + c.r / 4, c.y);
                    ctx.lineTo(c.x + c.tilt, c.y + c.tilt + c.r / 4);
                    ctx.stroke();
                    
                    c.y += c.speed;
                    c.tilt += c.tiltAngle;
                    c.tiltAngle += c.tiltAngleIncrement;
                    c.x += Math.sin(frame * 0.01 + i) * 2;
                    
                    if (c.y > canvas.height) {{
                        c.y = -10;
                        c.x = Math.random() * canvas.width;
                    }}
                }});
                
                frame++;
                if (frame < maxFrames) {{
                    requestAnimationFrame(draw);
                }} else {{
                    canvas.remove();
                }}
            }}
            
            draw();
        }})();
        </script>
        """
        components.html(html, height=0)
    
    @staticmethod
    def progress_ring(
        progress: float,
        size: int = 120,
        stroke_width: int = 10,
        color: str = "#4F46E5",
        bg_color: str = "#E2E8F0",
        label: str = ""
    ):
        """
        Display an animated circular progress ring.
        
        Args:
            progress: Progress value (0-100)
            size: Size of the ring in pixels
            stroke_width: Width of the progress stroke
            color: Color of the progress
            bg_color: Background color of the ring
            label: Optional label in the center
        """
        radius = (size - stroke_width) / 2
        circumference = radius * 2 * 3.14159
        offset = circumference - (progress / 100) * circumference
        
        html = f"""
        <div style="display: flex; justify-content: center; align-items: center; padding: 20px;">
            <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
                <circle
                    cx="{size/2}"
                    cy="{size/2}"
                    r="{radius}"
                    stroke="{bg_color}"
                    stroke-width="{stroke_width}"
                    fill="none"
                />
                <circle
                    id="progress-ring"
                    cx="{size/2}"
                    cy="{size/2}"
                    r="{radius}"
                    stroke="{color}"
                    stroke-width="{stroke_width}"
                    fill="none"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{circumference}"
                    style="transition: stroke-dashoffset 1.5s ease-out;"
                />
            </svg>
            <div style="position: absolute; font-size: 1.5rem; font-weight: 700; color: {color};">
                {progress:.1f}%
                {f'<div style="font-size: 0.75rem; color: #64748B; font-weight: 400;">{label}</div>' if label else ''}
            </div>
        </div>
        <script>
            setTimeout(() => {{
                document.getElementById('progress-ring').style.strokeDashoffset = '{offset}';
            }}, 100);
        </script>
        """
        components.html(html, height=size + 40)
    
    @staticmethod
    def typing_effect(text: str, speed: int = 50, style: str = ""):
        """
        Display text with a typewriter animation effect.
        
        Args:
            text: Text to display
            speed: Typing speed in milliseconds per character
            style: Additional CSS styles
        """
        html = f"""
        <div id="typing-container" style="{style}"></div>
        <script>
            (function() {{
                const text = `{text}`;
                const container = document.getElementById('typing-container');
                let i = 0;
                
                function type() {{
                    if (i < text.length) {{
                        container.innerHTML += text.charAt(i);
                        i++;
                        setTimeout(type, {speed});
                    }}
                }}
                
                type();
            }})();
        </script>
        """
        components.html(html, height=100)
    
    @staticmethod
    def shimmer_loading(height: int = 60, border_radius: int = 8):
        """
        Display a shimmer loading placeholder.
        
        Args:
            height: Height of the loading placeholder
            border_radius: Border radius in pixels
        """
        html = f"""
        <div class="shimmer" style="
            height: {height}px;
            border-radius: {border_radius}px;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        "></div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """
        components.html(html, height=height + 10)
    
    @staticmethod
    def particle_background(
        particle_count: int = 50,
        colors: List[str] = None,
        speed: float = 1.0
    ):
        """
        Add floating particles to the background.
        
        Args:
            particle_count: Number of particles
            colors: List of particle colors
            speed: Animation speed multiplier
        """
        if colors is None:
            colors = ["#4F46E5", "#7C3AED", "#10B981"]
        
        colors_js = json.dumps(colors)
        
        html = f"""
        <canvas id="particles-canvas" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: -1; opacity: 0.3;"></canvas>
        <script>
        (function() {{
            const canvas = document.getElementById('particles-canvas');
            const ctx = canvas.getContext('2d');
            
            function resize() {{
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }}
            resize();
            window.addEventListener('resize', resize);
            
            const colors = {colors_js};
            const particles = [];
            const count = {particle_count};
            const speed = {speed};
            
            for (let i = 0; i < count; i++) {{
                particles.push({{
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    r: Math.random() * 3 + 1,
                    color: colors[Math.floor(Math.random() * colors.length)],
                    vx: (Math.random() - 0.5) * speed,
                    vy: (Math.random() - 0.5) * speed
                }});
            }}
            
            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                particles.forEach(p => {{
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = p.color;
                    ctx.fill();
                    
                    p.x += p.vx;
                    p.y += p.vy;
                    
                    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                }});
                
                // Draw connections
                particles.forEach((p1, i) => {{
                    particles.slice(i + 1).forEach(p2 => {{
                        const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
                        if (dist < 100) {{
                            ctx.beginPath();
                            ctx.moveTo(p1.x, p1.y);
                            ctx.lineTo(p2.x, p2.y);
                            ctx.strokeStyle = `rgba(79, 70, 229, ${{1 - dist / 100}})`;
                            ctx.lineWidth = 0.5;
                            ctx.stroke();
                        }}
                    }});
                }});
                
                requestAnimationFrame(draw);
            }}
            
            draw();
        }})();
        </script>
        """
        components.html(html, height=0)
    
    @staticmethod
    def success_checkmark(size: int = 80, color: str = "#22C55E"):
        """
        Display an animated success checkmark.
        
        Args:
            size: Size of the checkmark in pixels
            color: Color of the checkmark
        """
        html = f"""
        <div style="display: flex; justify-content: center; padding: 20px;">
            <svg width="{size}" height="{size}" viewBox="0 0 52 52">
                <circle class="checkmark-circle" cx="26" cy="26" r="25" fill="none" stroke="{color}" stroke-width="2"/>
                <path class="checkmark-check" fill="none" stroke="{color}" stroke-width="4" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
            </svg>
        </div>
        <style>
            .checkmark-circle {{
                stroke-dasharray: 166;
                stroke-dashoffset: 166;
                animation: stroke 0.6s cubic-bezier(0.65, 0, 0.45, 1) forwards;
            }}
            .checkmark-check {{
                stroke-dasharray: 48;
                stroke-dashoffset: 48;
                animation: stroke 0.3s cubic-bezier(0.65, 0, 0.45, 1) 0.6s forwards;
            }}
            @keyframes stroke {{
                100% {{ stroke-dashoffset: 0; }}
            }}
        </style>
        """
        components.html(html, height=size + 40)
    
    @staticmethod
    def morphing_blob(
        colors: List[str] = None,
        size: int = 200
    ):
        """
        Display a morphing blob animation.
        
        Args:
            colors: Gradient colors
            size: Size of the blob
        """
        if colors is None:
            colors = ["#4F46E5", "#7C3AED", "#EC4899"]
        
        html = f"""
        <div style="display: flex; justify-content: center; padding: 20px;">
            <div class="blob" style="
                width: {size}px;
                height: {size}px;
                background: linear-gradient(135deg, {colors[0]}, {colors[1]}, {colors[2]});
                border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
                animation: morph 8s ease-in-out infinite;
            "></div>
        </div>
        <style>
            @keyframes morph {{
                0% {{ border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }}
                25% {{ border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }}
                50% {{ border-radius: 50% 60% 30% 60% / 30% 60% 70% 40%; }}
                75% {{ border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }}
                100% {{ border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }}
            }}
        </style>
        """
        components.html(html, height=size + 40)


class LottieAnimations:
    """
    Lottie animation integration for high-quality vector animations.
    """
    
    ANIMATIONS = {
        'loading': 'https://assets3.lottiefiles.com/packages/lf20_usmfx6bp.json',
        'success': 'https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json',
        'error': 'https://assets4.lottiefiles.com/packages/lf20_qpwbiyxf.json',
        'processing': 'https://assets2.lottiefiles.com/packages/lf20_szlepvdh.json',
        'data': 'https://assets8.lottiefiles.com/packages/lf20_qxzqzr7m.json',
        'rocket': 'https://assets3.lottiefiles.com/packages/lf20_l3qxn9jy.json',
    }
    
    @staticmethod
    def render(
        animation_name: str = 'loading',
        url: Optional[str] = None,
        height: int = 200,
        loop: bool = True,
        autoplay: bool = True
    ):
        """
        Render a Lottie animation.
        
        Args:
            animation_name: Name from ANIMATIONS dict
            url: Custom animation URL (overrides animation_name)
            height: Height of the animation
            loop: Whether to loop the animation
            autoplay: Whether to autoplay the animation
        """
        anim_url = url or LottieAnimations.ANIMATIONS.get(animation_name, LottieAnimations.ANIMATIONS['loading'])
        
        html = f"""
        <div id="lottie-container" style="height: {height}px; display: flex; justify-content: center;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
        <script>
            lottie.loadAnimation({{
                container: document.getElementById('lottie-container'),
                renderer: 'svg',
                loop: {str(loop).lower()},
                autoplay: {str(autoplay).lower()},
                path: '{anim_url}'
            }});
        </script>
        """
        components.html(html, height=height)


class GlassmorphicCards:
    """
    Glassmorphism style cards with blur effects.
    """
    
    @staticmethod
    def metric_card(
        title: str,
        value: str,
        delta: Optional[str] = None,
        delta_color: str = "green",
        icon: str = "📊"
    ):
        """
        Display a glassmorphic metric card.
        
        Args:
            title: Card title
            value: Main value to display
            delta: Optional delta/change indicator
            delta_color: Color for delta (green/red)
            icon: Emoji icon
        """
        delta_html = ""
        if delta:
            color = "#22C55E" if delta_color == "green" else "#EF4444"
            arrow = "↑" if delta_color == "green" else "↓"
            delta_html = f'<div style="color: {color}; font-size: 0.875rem;">{arrow} {delta}</div>'
        
        html = f"""
        <div class="glass-card" style="
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 24px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
            transition: all 0.3s ease;
        ">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 2rem;">{icon}</span>
                <span style="color: #64748B; font-size: 0.875rem; font-weight: 500;">{title}</span>
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #1E293B;">{value}</div>
            {delta_html}
        </div>
        <style>
            .glass-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25);
            }}
        </style>
        """
        components.html(html, height=160)
    
    @staticmethod
    def info_card(
        title: str,
        content: str,
        accent_color: str = "#4F46E5"
    ):
        """
        Display a glassmorphic info card.
        
        Args:
            title: Card title
            content: Card content
            accent_color: Accent color for the top border
        """
        html = f"""
        <div style="
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-top: 4px solid {accent_color};
            padding: 20px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        ">
            <h4 style="margin: 0 0 12px 0; color: #1E293B; font-weight: 600;">{title}</h4>
            <p style="margin: 0; color: #64748B; font-size: 0.9rem; line-height: 1.6;">{content}</p>
        </div>
        """
        components.html(html, height=150)


class InteractiveChartEnhancements:
    """
    Enhanced chart interactions and animations.
    """
    
    @staticmethod
    def get_modern_plotly_theme() -> Dict[str, Any]:
        """
        Get a modern Plotly theme configuration.
        
        Returns:
            Dict with Plotly layout configuration
        """
        return {
            'template': 'plotly_white',
            'font': {
                'family': 'Inter, sans-serif',
                'size': 12,
                'color': '#334155'
            },
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'colorway': [
                '#4F46E5', '#7C3AED', '#EC4899', '#10B981',
                '#F59E0B', '#06B6D4', '#8B5CF6', '#EF4444'
            ],
            'hoverlabel': {
                'bgcolor': 'white',
                'font_size': 13,
                'font_family': 'Inter, sans-serif',
                'bordercolor': '#E2E8F0'
            },
            'legend': {
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': '#E2E8F0',
                'borderwidth': 1,
                'font': {'size': 11}
            },
            'xaxis': {
                'gridcolor': '#F1F5F9',
                'linecolor': '#E2E8F0',
                'tickfont': {'size': 11}
            },
            'yaxis': {
                'gridcolor': '#F1F5F9',
                'linecolor': '#E2E8F0',
                'tickfont': {'size': 11}
            }
        }
    
    @staticmethod
    def add_animation_config() -> Dict[str, Any]:
        """
        Get animation configuration for Plotly charts.
        
        Returns:
            Dict with animation configuration
        """
        return {
            'transition': {
                'duration': 500,
                'easing': 'cubic-in-out'
            },
            'frame': {
                'duration': 500,
                'redraw': True
            }
        }
    
    @staticmethod
    def get_gradient_colorscale(
        start_color: str = "#4F46E5",
        end_color: str = "#EC4899"
    ) -> List[List]:
        """
        Get a custom gradient colorscale.
        
        Args:
            start_color: Starting color
            end_color: Ending color
            
        Returns:
            Plotly-compatible colorscale
        """
        return [
            [0, start_color],
            [0.5, "#7C3AED"],
            [1, end_color]
        ]
