"""
Cinematic 3D Medical Laboratory Hero Component.
===============================================
Renders an interactive WebGL / Three.js clinical visualization:
  - 3D floating clinical monitors (Thoracic Scan, ECG Pulse, Telemetry Stream, Neural Attention).
  - Dynamic interconnecting synapto-telemetry data beams.
  - Smooth interactive mouse parallax depth.
  - Controlled medical studio lighting (white key light, cyan accent rim light, deep navy environment).
"""

from __future__ import annotations

import streamlit.components.v1 as components


def render_cinematic_3d_hero(height: int = 560) -> None:
    """
    Renders the responsive 3D WebGL medical laboratory hero viewport.
    """
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: #090D16;
                overflow: hidden;
                font-family: 'Inter', -apple-system, sans-serif;
                width: 100vw;
                height: 100vh;
            }
            #canvas-container {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
            }
            .hero-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 2;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                background: radial-gradient(circle at 50% 50%, rgba(9, 13, 22, 0.2) 0%, rgba(9, 13, 22, 0.75) 100%);
                pointer-events: none;
                padding: 20px;
            }
            .badge-live {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(8, 145, 178, 0.15);
                border: 1px solid rgba(8, 145, 178, 0.4);
                color: #22D3EE;
                padding: 6px 16px;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 16px;
                backdrop-filter: blur(8px);
                animation: pulse-glow 3s infinite;
            }
            @keyframes pulse-glow {
                0%, 100% { border-color: rgba(8, 145, 178, 0.4); box-shadow: 0 0 10px rgba(8, 145, 178, 0.2); }
                50% { border-color: rgba(37, 99, 235, 0.8); box-shadow: 0 0 20px rgba(37, 99, 235, 0.4); }
            }
            .title-hero {
                font-size: 2.85rem;
                font-weight: 800;
                color: #FFFFFF;
                letter-spacing: -0.03em;
                line-height: 1.15;
                margin-bottom: 12px;
                text-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
            }
            .title-gradient {
                background: linear-gradient(135deg, #FFFFFF 30%, #93C5FD 70%, #38BDF8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .tagline-hero {
                font-size: 1.15rem;
                color: #94A3B8;
                font-weight: 400;
                max-width: 680px;
                line-height: 1.5;
                margin-bottom: 28px;
                letter-spacing: -0.01em;
            }
            .hud-metrics {
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }
            .hud-item {
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 8px;
                padding: 8px 16px;
                backdrop-filter: blur(6px);
            }
            .hud-num {
                font-size: 1.1rem;
                font-weight: 700;
                color: #38BDF8;
                font-family: monospace;
            }
            .hud-label {
                font-size: 0.65rem;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="canvas-container"></div>
        <div class="hero-overlay">
            <div class="badge-live">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #22D3EE; box-shadow: 0 0 8px #22D3EE;"></span>
                Autonomous Emergency Intelligence &bull; Active Clinical Model
            </div>
            <h1 class="title-hero">
                Multimodal Intelligent<br/><span class="title-gradient">Triage Assistant</span>
            </h1>
            <p class="tagline-hero">
                &ldquo;Where Clinical Intelligence Meets Multimodal AI.&rdquo;
            </p>
            <div class="hud-metrics">
                <div class="hud-item">
                    <div class="hud-num">MiniLM + Whisper</div>
                    <div class="hud-label">Speech & NLP</div>
                </div>
                <div class="hud-item">
                    <div class="hud-num">DenseNet-121</div>
                    <div class="hud-label">CheXNet Radiography</div>
                </div>
                <div class="hud-item">
                    <div class="hud-num">CECE v2</div>
                    <div class="hud-label">Cross-Modal Audit</div>
                </div>
                <div class="hud-item">
                    <div class="hud-num">ECE 0.067</div>
                    <div class="hud-label">Calibrated Safety</div>
                </div>
            </div>
        </div>

        <script>
            // Three.js 3D Scene Initialization
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x090D16, 0.035);

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 18);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            // Studio Lighting
            const ambientLight = new THREE.AmbientLight(0x1E293B, 1.5);
            scene.add(ambientLight);

            const blueLight = new THREE.PointLight(0x2563EB, 3.5, 50);
            blueLight.position.set(-15, 10, 10);
            scene.add(blueLight);

            const cyanLight = new THREE.PointLight(0x0891B2, 3.0, 50);
            cyanLight.position.set(15, -10, 12);
            scene.add(cyanLight);

            // Ground Telemetry Grid
            const gridHelper = new THREE.GridHelper(60, 40, 0x1E293B, 0x0F172A);
            gridHelper.position.y = -8;
            scene.add(gridHelper);

            // Floating Clinical Hologram Panels
            const panelsGroup = new THREE.Group();
            scene.add(panelsGroup);

            function createHoloPanel(x, y, z, rotY, color, labelText) {
                const group = new THREE.Group();
                group.position.set(x, y, z);
                group.rotation.y = rotY;

                // Border wireframe
                const geom = new THREE.PlaneGeometry(5, 3.2);
                const edges = new THREE.EdgesGeometry(geom);
                const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 }));
                group.add(line);

                // Semi-transparent surface
                const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color: 0x0F172A, transparent: true, opacity: 0.45 }));
                group.add(mesh);

                // Corner brackets
                const bracketGeom = new THREE.BufferGeometry();
                const bSize = 0.5;
                const bPts = [
                    new THREE.Vector3(-2.5, 1.6 - bSize, 0.01), new THREE.Vector3(-2.5, 1.6, 0.01),
                    new THREE.Vector3(-2.5, 1.6, 0.01), new THREE.Vector3(-2.5 + bSize, 1.6, 0.01),
                    new THREE.Vector3(2.5 - bSize, 1.6, 0.01), new THREE.Vector3(2.5, 1.6, 0.01),
                    new THREE.Vector3(2.5, 1.6, 0.01), new THREE.Vector3(2.5, 1.6 - bSize, 0.01),
                ];
                bracketGeom.setFromPoints(bPts);
                const bracketLine = new THREE.LineSegments(bracketGeom, new THREE.LineBasicMaterial({ color: color }));
                group.add(bracketLine);

                return group;
            }

            const p1 = createHoloPanel(-9.5, 1.5, -4, 0.35, 0x38BDF8, "PULSE");
            const p2 = createHoloPanel(9.5, 1.2, -4, -0.35, 0x3B82F6, "XRAY");
            const p3 = createHoloPanel(-6.0, -3.8, 1, 0.15, 0x06B6D4, "NLP");
            const p4 = createHoloPanel(6.0, -3.8, 1, -0.15, 0x10B981, "KTAS");

            panelsGroup.add(p1);
            panelsGroup.add(p2);
            panelsGroup.add(p3);
            panelsGroup.add(p4);

            // Animated Data Connection Lines
            const connGeom = new THREE.BufferGeometry();
            const connPts = [
                p1.position, p3.position,
                p3.position, p4.position,
                p4.position, p2.position,
                p1.position, p2.position
            ];
            connGeom.setFromPoints(connPts);
            const connLines = new THREE.LineSegments(connGeom, new THREE.LineBasicMaterial({ color: 0x0891B2, transparent: true, opacity: 0.25 }));
            scene.add(connLines);

            // Floating Neural Particle Cloud
            const particleCount = 280;
            const particleGeom = new THREE.BufferGeometry();
            const posArray = new Float32Array(particleCount * 3);

            for (let i = 0; i < particleCount * 3; i += 3) {
                posArray[i] = (Math.random() - 0.5) * 35;
                posArray[i + 1] = (Math.random() - 0.5) * 20;
                posArray[i + 2] = (Math.random() - 0.5) * 20;
            }
            particleGeom.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            const particleMat = new THREE.PointsMaterial({ size: 0.08, color: 0x38BDF8, transparent: true, opacity: 0.65 });
            const particles = new THREE.Points(particleGeom, particleMat);
            scene.add(particles);

            // Subtle Mouse Parallax
            let mouseX = 0;
            let mouseY = 0;
            let targetX = 0;
            let targetY = 0;
            const windowHalfX = window.innerWidth / 2;
            const windowHalfY = window.innerHeight / 2;

            document.addEventListener('mousemove', (event) => {
                mouseX = (event.clientX - windowHalfX) * 0.0008;
                mouseY = (event.clientY - windowHalfY) * 0.0008;
            });

            // Animation Loop
            let clock = new THREE.Clock();

            function animate() {
                requestAnimationFrame(animate);
                const elapsedTime = clock.getElapsedTime();

                // Smooth camera parallax
                targetX += (mouseX - targetX) * 0.05;
                targetY += (mouseY - targetY) * 0.05;
                camera.position.x = targetX * 4;
                camera.position.y = -targetY * 3;
                camera.lookAt(0, 0, 0);

                // Gentle floating animation of panels
                p1.position.y = 1.5 + Math.sin(elapsedTime * 1.2) * 0.25;
                p2.position.y = 1.2 + Math.cos(elapsedTime * 1.1) * 0.25;
                p3.position.y = -3.8 + Math.sin(elapsedTime * 1.3 + 1) * 0.20;
                p4.position.y = -3.8 + Math.cos(elapsedTime * 1.2 + 2) * 0.20;

                // Subtle grid sweep
                gridHelper.position.z = (elapsedTime * 0.8) % 1.5;

                // Particle slow rotation
                particles.rotation.y = elapsedTime * 0.02;

                renderer.render(scene, camera);
            }
            animate();

            // Resize Handler
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)

