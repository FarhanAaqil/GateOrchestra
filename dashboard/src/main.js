const root = document.getElementById("root");

root.innerHTML = `
    <div class="dashboard">

        <header class="header">
            <h1>GateOrchestra</h1>
            <p>Token-Budget-Calibrated Multi-Agent Orchestration</p>
        </header>

        <main>

            <section class="cards">

                <div class="card">
                    <h3>System Status</h3>
                    <div class="value online">ONLINE</div>
                </div>

                <div class="card">
                    <h3>Token Budget</h3>
                    <div class="value">10,000</div>
                </div>

                <div class="card">
                    <h3>Agents</h3>
                    <div class="value">4</div>
                </div>

                <div class="card">
                    <h3>Current Gate</h3>
                    <div class="value">AUTO</div>
                </div>

            </section>

            <section class="panel">
                <h2>Orchestration Dashboard</h2>

                <p>
                    GateOrchestra dynamically decides which agents
                    should participate based on the available token budget.
                </p>

                <p>
                    <strong>Current Strategy:</strong>
                    <span class="online"> Adaptive Gating</span>
                </p>
            </section>

            <section class="panel">
                <h2>Agent Pipeline</h2>

                <div class="agent">
                    🧠 Reasoning Agent
                </div>

                <div class="agent">
                    🔍 Verification Agent
                </div>

                <div class="agent">
                    📊 Analysis Agent
                </div>

                <div class="agent">
                    ⚡ Final Response Agent
                </div>

            </section>

        </main>

    </div>
`;

const style = document.createElement("style");

style.textContent = `
    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #0f172a;
        color: white;
    }

    .dashboard {
        min-height: 100vh;
    }

    .header {
        padding: 25px 35px;
        background: #1e293b;
        border-bottom: 1px solid #334155;
    }

    .header h1 {
        margin: 0;
        font-size: 30px;
    }

    .header p {
        margin-top: 8px;
        color: #94a3b8;
    }

    main {
        padding: 35px;
    }

    .cards {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
    }

    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
    }

    .card h3 {
        margin: 0 0 15px;
        color: #94a3b8;
    }

    .value {
        font-size: 30px;
        font-weight: bold;
    }

    .online {
        color: #22c55e;
    }

    .panel {
        margin-top: 25px;
        padding: 25px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
    }

    .panel h2 {
        margin-top: 0;
    }

    .agent {
        padding: 15px;
        margin-top: 10px;
        background: #0f172a;
        border-radius: 8px;
        border: 1px solid #334155;
    }

    @media (max-width: 900px) {
        .cards {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 600px) {
        .cards {
            grid-template-columns: 1fr;
        }

        main {
            padding: 20px;
        }
    }
`;

document.head.appendChild(style);