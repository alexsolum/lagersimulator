import streamlit as st
import simpy
import pandas as pd
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from ortools.linear_solver import pywraplp
import io

# --- KONFIGURASJON ---
st.set_page_config(page_title="LagerSimulering AI", layout="wide")

# --- HJELPEFUNKSJONER FOR GRAF OG RUTING ---

def bygg_lager_graf(layout_df):
    """
    Bygger en retet graf (DiGraph) basert på layout.
    Håndterer hindringer (Reoler) og enveiskjøring.
    """
    G = nx.DiGraph()
    
    # 1. Opprett noder for alle koordinater i et grid som dekker lageret
    max_x = layout_df["X"].max() + 2
    max_y = layout_df["Y"].max() + 2
    
    # Lag et oppslagsverk for layout-typer på koordinater
    grid_map = {}
    for _, row in layout_df.iterrows():
        grid_map[(row["X"], row["Y"])] = row
        
    # Legg til noder og kanter
    for x in range(max_x):
        for y in range(max_y):
            current_pos = (x, y)
            
            # Sjekk om dette er en hindring
            info = grid_map.get(current_pos)
            if info is not None and info["Type"] == "Reol":
                continue # Kan ikke gå her (hindring)
            
            G.add_node(current_pos)
            
            # Sjekk naboer (Nord, Sør, Øst, Vest)
            directions = [
                (0, 1, 'N'), (0, -1, 'S'), 
                (1, 0, 'E'), (-1, 0, 'W')
            ]
            
            # Bestem tillatte retninger basert på enveiskjøring
            allowed_dirs = ['N', 'S', 'E', 'W']
            if info is not None and pd.notna(info.get("Retning")):
                # Hvis Retning er definert (f.eks. 'N'), tillat kun den retningen
                allowed_dirs = [info["Retning"]]
            
            for dx, dy, dir_code in directions:
                if dir_code not in allowed_dirs:
                    continue
                    
                neighbor = (x + dx, y + dy)
                
                # Sjekk om nabo er innafor grid
                if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y:
                    # Sjekk om nabo er hindring
                    n_info = grid_map.get(neighbor)
                    if n_info is not None and n_info["Type"] == "Reol":
                        continue
                    
                    # Legg til kant (vei)
                    # Vekting: Kan legge til 'weight' her hvis vi vil prioritere visse veier
                    G.add_edge(current_pos, neighbor, weight=1)
    
    return G

def finn_korteste_rute(G, start_pos, end_pos):
    """Finner korteste vei i grafen. Returnerer lengde og sti."""
    try:
        path = nx.shortest_path(G, source=start_pos, target=end_pos, weight='weight')
        # Avstand er antall steg (kanter)
        dist = len(path) - 1 
        return dist, path
    except nx.NetworkXNoPath:
        return float('inf'), []
    except nx.NodeNotFound:
        # Fallback hvis noden er "inni" en reol pga dårlig data: bruk Manhattan
        return abs(start_pos[0]-end_pos[0]) + abs(start_pos[1]-end_pos[1]), []

def convert_df_to_excel(df):
    """Hjelpefunksjon for å konvertere DataFrame til Excel-bytes for nedlasting."""
    output = io.BytesIO()
    # Bruker xlsxwriter som motor (standard i mange miljøer)
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- DATA GENERERING ---

def generer_avansert_layout():
    data = []
    # Lageret er 20 bredt, 30 høyt
    # Vi lager reoler som blokkerer, med ganger imellom
    
    # Gang i midten og rundt
    for y in range(30):
        for x in range(20):
            pos = {"LocationID": f"POS-{x}-{y}", "X": x, "Y": y, "Type": "Gang", "Retning": None}
            
            # Definer Reoler (Hindringer)
            if x in [2, 3, 6, 7, 10, 11, 14, 15] and 2 < y < 28:
                if y != 15: # Tverrgang på midten
                    pos["Type"] = "Reol" # Hindring
            
            # Definer Plukk-lokasjoner (ved siden av reolene i gangen)
            # Vi plasserer varene i gangen rett utenfor reolen for simuleringens del
            if x in [1, 4, 5, 8, 9, 12, 13, 16] and 2 < y < 28:
                 if y != 15:
                    pos["Type"] = "Plukk"
                    pos["LocationID"] = f"LOC-{x}-{y}"
            
            # Enveiskjøring i ganger mellom reoler
            if pos["Type"] in ["Gang", "Plukk"]:
                if x == 1 and 2 < y < 28: pos["Retning"] = "N" # Opp
                if x == 4 and 2 < y < 28: pos["Retning"] = "S" # Ned
                if x == 5 and 2 < y < 28: pos["Retning"] = "N"
                if x == 8 and 2 < y < 28: pos["Retning"] = "S"
                # ... osv

            data.append(pos)

    # Spesielle lokasjoner
    # Overskriv eksisterende punkter
    spesielle = [
        {"LocationID": "START", "X": 0, "Y": 0, "Type": "Start"},
        {"LocationID": "PRINTER", "X": 0, "Y": 15, "Type": "Printer"},
        {"LocationID": "STAGING", "X": 0, "Y": 29, "Type": "Oppstilling"}
    ]
    
    df = pd.DataFrame(data)
    # Fjern de punktene vi overskriver
    for s in spesielle:
        mask = (df["X"] == s["X"]) & (df["Y"] == s["Y"])
        df = df[~mask]
    
    df = pd.concat([df, pd.DataFrame(spesielle)], ignore_index=True)
    
    # Tildel tilfeldige artikler til plukkplasser
    plukk_mask = df["Type"] == "Plukk"
    antall_plukk = plukk_mask.sum()
    artikler = [f"ART-{i}" for i in range(1000, 1000 + antall_plukk)]
    df.loc[plukk_mask, "ArtikkelID"] = artikler[:antall_plukk]
    
    return df

def generer_ordre_med_historikk(layout_df, antall_ordre=50):
    plukk_df = layout_df[layout_df["Type"] == "Plukk"]
    if plukk_df.empty: return pd.DataFrame()
    
    artikler = plukk_df["ArtikkelID"].dropna().unique()
    
    ordre_data = []
    # Lag noen "populære" varer (Pareto: 20% av varene i 80% av ordrene)
    num_hot = int(len(artikler) * 0.2)
    hot_items = artikler[:num_hot] if num_hot > 0 else artikler
    cold_items = artikler[num_hot:] if num_hot > 0 else []

    for i in range(1, antall_ordre + 1):
        antall_linjer = random.randint(1, 8)
        ordre_start = random.randint(0, 120) # Ordre kommer inn over 2 timer
        
        for _ in range(antall_linjer):
            # 80% sjanse for hot item
            if hot_items.size > 0 and random.random() < 0.8:
                valgt_artikkel = np.random.choice(hot_items)
            elif cold_items.size > 0:
                valgt_artikkel = np.random.choice(cold_items)
            else:
                valgt_artikkel = np.random.choice(artikler)

            # Finn lokasjon for artikkelen
            loc_row = layout_df[layout_df["ArtikkelID"] == valgt_artikkel]
            if not loc_row.empty:
                loc_id = loc_row.iloc[0]["LocationID"]
                ordre_data.append({
                    "OrdreID": f"ORD-{i}",
                    "ArtikkelID": valgt_artikkel,
                    "Antall": random.randint(1, 5),
                    "LocationID": loc_id,
                    "TidligstStart": ordre_start
                })
    return pd.DataFrame(ordre_data)

# --- SIMPY MODELL ---

class LagerSimulering:
    def __init__(self, env, layout_df, ordre_df, config):
        self.env = env
        self.layout = layout_df.set_index("LocationID")
        self.ordre_liste = ordre_df.groupby("OrdreID")
        self.config = config
        
        # Bygg graf for ruting
        self.G = bygg_lager_graf(layout_df)
        
        # Ressurser
        self.plukkere = simpy.Resource(env, capacity=config['antall_plukkere'])
        self.printer = simpy.Resource(env, capacity=1) # Flaskehals
        
        # Plukklokasjoner som ressurser (hvis vi vil simulere kø i gangen/ved vare)
        # Lager en ressurs per plukklokasjon med kapasitet 1
        self.loc_resources = {
            loc_id: simpy.Resource(env, capacity=1) 
            for loc_id in self.layout.index if self.layout.loc[loc_id, "Type"] == "Plukk"
        }

        # Statistikk
        self.logg = []
        self.ferdig_ordre = 0
        self.ordre_queue_times = []

    def get_coords(self, loc_id):
        if loc_id in self.layout.index:
            return (self.layout.loc[loc_id, "X"], self.layout.loc[loc_id, "Y"])
        # Fallback let etter Type
        res = self.layout[self.layout["Type"] == loc_id]
        if not res.empty:
            return (res.iloc[0]["X"], res.iloc[0]["Y"])
        return (0, 0)

    def move_to(self, current_pos, target_loc_id, plukker_navn, ordre_id):
        """Hjelpefunksjon for bevegelse med graf"""
        target_pos = self.get_coords(target_loc_id)
        dist, _ = finn_korteste_rute(self.G, current_pos, target_pos)
        
        if dist == float('inf'):
            # Fallback hvis graf feiler (f.eks. ugyldig start/slutt)
            dist = abs(current_pos[0]-target_pos[0]) + abs(current_pos[1]-target_pos[1])
            
        reise_tid = (dist / self.config['gangfart']) * 60 # sekunder til minutter? La oss si gangfart er ruter pr minutt
        # Hvis gangfart er m/min og en rute er 1m:
        tid = dist / self.config['gangfart']
        
        yield self.env.timeout(tid)
        return target_pos

    def plukker_prosess(self, ordre_id, linjer):
        # 1. Vent på TidligstStart
        start_tid = linjer["TidligstStart"].min()
        if self.env.now < start_tid:
            yield self.env.timeout(start_tid - self.env.now)

        queue_start = self.env.now
        # 2. Be om plukker (ressurs)
        with self.plukkere.request() as req:
            yield req
            ventetid = self.env.now - queue_start
            self.ordre_queue_times.append(ventetid)
            
            plukker_navn = f"P-{random.randint(1, self.config['antall_plukkere'])}"
            
            # PAUSE LOGIKK
            # Enkel sjekk: Hvis klokka er f.eks. > 120 min (2 timer) og plukker ikke har hatt pause... 
            # For simulering: 5% sjanse for pause mellom hver ordre
            if self.config['simuler_pauser'] and random.random() < 0.05:
                self.log_event(self.env.now, ordre_id, f"{plukker_navn} tar Pause", (0,0))
                yield self.env.timeout(15) # 15 min pause
            
            start_pos = self.get_coords("START")
            curr_pos = start_pos
            self.log_event(self.env.now, ordre_id, f"Start ({plukker_navn})", curr_pos)

            # 3. Plukkrunde
            for _, rad in linjer.iterrows():
                dest_loc = rad["LocationID"]
                
                # Gå til vare
                curr_pos = yield from self.move_to(curr_pos, dest_loc, plukker_navn, ordre_id)
                self.log_event(self.env.now, ordre_id, f"Ankomst {dest_loc}", curr_pos)
                
                # Simulere kø ved plukkplass/gang?
                loc_res = self.loc_resources.get(dest_loc)
                if loc_res:
                    with loc_res.request() as loc_req:
                        yield loc_req # Vent hvis noen andre står akkurat der
                        
                        # Plukke (Tid)
                        basis = 60 / self.config['dpak_time']
                        var = random.uniform(1-self.config['varians'], 1+self.config['varians'])
                        plukk_tid = max(0.1, basis * var * rad["Antall"])
                        yield self.env.timeout(plukk_tid)
                        self.log_event(self.env.now, ordre_id, f"Plukket {rad['ArtikkelID']}", curr_pos)
                else:
                    # Fallback hvis ressurs mangler
                    yield self.env.timeout(0.5)

            # 4. Gå til Printer (Kø-simulering)
            curr_pos = yield from self.move_to(curr_pos, "PRINTER", plukker_navn, ordre_id)
            
            # PRINTER KØ
            print_queue_start = self.env.now
            with self.printer.request() as print_req:
                if len(self.printer.queue) > 0:
                     self.log_event(self.env.now, ordre_id, "Kø ved Printer", curr_pos)
                yield print_req
                # Hvis vi ventet, logg det
                if self.env.now - print_queue_start > 0.1:
                    self.log_event(self.env.now, ordre_id, "Startet Print", curr_pos)
                
                yield self.env.timeout(0.8) # Tid ved printer
            
            self.log_event(self.env.now, ordre_id, "Ferdig Printer", curr_pos)

            # 5. Gå til Oppstilling
            curr_pos = yield from self.move_to(curr_pos, "STAGING", plukker_navn, ordre_id)
            self.log_event(self.env.now, ordre_id, "Levert Staging", curr_pos)
            
            self.ferdig_ordre += 1

    def log_event(self, tid, ordre, hendelse, pos):
        self.logg.append({
            "Tid": tid, "Ordre": ordre, "Hendelse": hendelse, "X": pos[0], "Y": pos[1]
        })

    def kjør(self):
        for ordre_id, linjer in self.ordre_liste:
            self.env.process(self.plukker_prosess(ordre_id, linjer))
        self.env.run()

# --- OPTIMALISERING (OR-TOOLS) ---

def optimaliser_vareplassering(layout_df, ordre_df):
    """
    Løser et Assignment Problem:
    Minimer total reiseavstand: Sum(Frekvens_i * Avstand_j)
    Hvor vare i plasseres på lokasjon j.
    """
    st.info("Starter optimalisering... Dette kan ta noen sekunder.")
    
    # 1. Beregn frekvens per artikkel
    item_counts = ordre_df["ArtikkelID"].value_counts()
    items = item_counts.index.tolist()
    
    # 2. Finn alle gyldige plukklokasjoner
    plukk_locs = layout_df[layout_df["Type"] == "Plukk"]["LocationID"].tolist()
    
    # Begrensning for demo: Ta kun topp N varer hvis det er flere varer enn plasser
    # Eller fyll opp tomme plasser.
    num_slots = len(plukk_locs)
    items_to_place = items[:num_slots] # Prioriter de mest populære
    
    if len(items_to_place) == 0 or num_slots == 0:
        return layout_df, 0.0

    # 3. Beregn "Kostnad" for hver lokasjon (Avstand fra Start + Avstand til Printer)
    # Dette er en forenkling. Ideelt sett bør vi se på samlokalisering (affinity), men det er kvadratisk kompleksitet.
    # Vi bruker "Avstand fra Start" som proxy for "Lett tilgjengelighet".
    
    # Bygg grafen en gang for avstandsberegning
    G = bygg_lager_graf(layout_df)
    start_pos = (layout_df[layout_df["Type"]=="Start"].iloc[0]["X"], layout_df[layout_df["Type"]=="Start"].iloc[0]["Y"])
    
    loc_costs = {}
    for loc in plukk_locs:
        row = layout_df[layout_df["LocationID"] == loc].iloc[0]
        pos = (row["X"], row["Y"])
        # Beregn avstand fra start til denne plassen
        dist, _ = finn_korteste_rute(G, start_pos, pos)
        loc_costs[loc] = dist

    # 4. OR-Tools Solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        st.error("Kunne ikke laste OR-Tools SCIP solver.")
        return layout_df, 0

    # Variabler: x[i, j] = 1 hvis item i er på lokasjon j
    x = {}
    for i in items_to_place:
        for j in plukk_locs:
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    # Constraint: Hvert item må ha én plass
    for i in items_to_place:
        solver.Add(solver.Sum([x[i, j] for j in plukk_locs]) == 1)

    # Constraint: Hver plass kan ha maks ett item
    for j in plukk_locs:
        solver.Add(solver.Sum([x[i, j] for i in items_to_place]) <= 1)

    # Objective: Minimer Sum(Frekvens * Kostnad)
    objective = solver.Objective()
    for i in items_to_place:
        freq = item_counts[i]
        for j in plukk_locs:
            cost = loc_costs[j]
            objective.SetCoefficient(x[i, j], freq * cost)
    
    objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        new_layout = layout_df.copy()
        # Nullstill eksisterende artikler på plukkplasser
        new_layout.loc[new_layout["Type"]=="Plukk", "ArtikkelID"] = None
        
        assigned_cost = objective.Value()
        
        # Oppdater layout med nye plasseringer
        for i in items_to_place:
            for j in plukk_locs:
                if x[i, j].solution_value() > 0.5:
                    # Sett artikkel i på plass j
                    idx = new_layout[new_layout["LocationID"] == j].index
                    new_layout.loc[idx, "ArtikkelID"] = i
        
        return new_layout, assigned_cost
    else:
        st.warning("Fant ingen optimal løsning.")
        return layout_df, 0


# --- UI LOGIKK ---

# Initialiser session state
if 'historikk' not in st.session_state: st.session_state['historikk'] = []
if 'layout_df' not in st.session_state: st.session_state['layout_df'] = None
if 'ordre_df' not in st.session_state: st.session_state['ordre_df'] = None

with st.sidebar:
    st.header("Input & Data")
    
    st.markdown("### 🛠️ Demo & Testing")
    # Knapper for generering
    if st.button("Generer Eksempeldata (Demo)", type="primary"):
        st.session_state['layout_df'] = generer_avansert_layout()
        st.session_state['ordre_df'] = generer_ordre_med_historikk(st.session_state['layout_df'])
        st.success("Demo-data lastet! Du kan nå kjøre simulering.")

    # Nedlasting av data for verifisering/testing
    if st.session_state.get('layout_df') is not None:
        st.markdown("---")
        st.markdown("**Last ned test-data:**")
        st.caption("Bruk disse filene som mal for egne oppsett.")
        
        excel_layout = convert_df_to_excel(st.session_state['layout_df'])
        st.download_button(
            label="📥 Last ned Layout (xlsx)",
            data=excel_layout,
            file_name="demo_layout.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        if st.session_state.get('ordre_df') is not None:
            excel_ordre = convert_df_to_excel(st.session_state['ordre_df'])
            st.download_button(
                label="📥 Last ned Ordre (xlsx)",
                data=excel_ordre,
                file_name="demo_ordre.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.markdown("---")

    st.subheader("Last opp egne data")
    uploaded_layout = st.file_uploader("Layout fil", type=["xlsx"])
    if uploaded_layout:
        st.session_state['layout_df'] = pd.read_excel(uploaded_layout)
        
    uploaded_ordre = st.file_uploader("Ordre fil", type=["xlsx"])
    if uploaded_ordre:
        st.session_state['ordre_df'] = pd.read_excel(uploaded_ordre)

    st.subheader("Konfigurasjon")
    cfg = {}
    cfg['antall_plukkere'] = st.number_input("Antall Plukkere", 1, 10, 3)
    cfg['dpak_time'] = st.number_input("Plukkhastighet (DPAK/t)", 50, 300, 120)
    cfg['varians'] = st.slider("Variasjon ansatte", 0.0, 0.5, 0.1)
    cfg['gangfart'] = st.number_input("Gangfart (ruter/min)", 10.0, 100.0, 40.0)
    cfg['simuler_pauser'] = st.checkbox("Simuler Pauser", True)

    start_knapp = st.button("▶️ Kjør Simulering", type="primary")

# HOVEDLOGIKK

layout_df = st.session_state['layout_df']
ordre_df = st.session_state['ordre_df']

if layout_df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Simulering", "🧠 Optimalisering", "🗺️ Layout Visning", "📋 Data"])
    
    with tab1:
        if start_knapp and ordre_df is not None:
            env = simpy.Environment()
            sim = LagerSimulering(env, layout_df, ordre_df, cfg)
            sim.kjør()
            
            # Lagre resultater
            logg_df = pd.DataFrame(sim.logg)
            total_tid = logg_df["Tid"].max()
            st.session_state['logg'] = logg_df
            
            # Beregn ventetid
            avg_wait = np.mean(sim.ordre_queue_times) if sim.ordre_queue_times else 0
            
            res = {
                "Plukkere": cfg['antall_plukkere'],
                "Tid (min)": round(total_tid, 1),
                "Ordre": sim.ferdig_ordre,
                "Snitt Ventetid": round(avg_wait, 1)
            }
            st.session_state['historikk'].append(res)
            
        # Dashboard
        if 'logg' in st.session_state:
            logg = st.session_state['logg']
            hist = st.session_state['historikk'][-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tid", f"{hist['Tid (min)']} min")
            col2.metric("Antall Ordre", hist['Ordre'])
            col3.metric("Snitt Ventetid (Ressurs)", f"{hist['Snitt Ventetid']} min")
            
            # Heatmap
            plukk_ev = logg[logg["Hendelse"].str.contains("Plukket")]
            if not plukk_ev.empty:
                fig = px.density_heatmap(plukk_ev, x="X", y="Y", title="Heatmap: Plukkaktivitet", nbinsx=20, nbinsy=30)
                st.plotly_chart(fig, use_container_width=True)
                
            # Kø ved Printer
            printer_ko = logg[logg["Hendelse"] == "Kø ved Printer"]
            if not printer_ko.empty:
                st.warning(f"Registrert kø ved printer {len(printer_ko)} ganger.")

    with tab2:
        st.markdown("### Vareplasserings-optimalisering")
        st.markdown("Bruker **OR-Tools** til å flytte varer med høy frekvens nærmere start og printer.")
        
        if st.button("Kjør Optimalisering"):
            ny_layout, kostnad = optimaliser_vareplassering(layout_df, ordre_df)
            st.session_state['layout_df'] = ny_layout
            st.success(f"Ny layout generert! Teoretisk kostnadsreduksjon kalkulert. Gå til 'Simulering' og kjør på nytt for å verifisere.")
            
            # Sammenlign før/etter (visuelt i tabell)
            st.write("Sjekk Layout Visning for å se endringer.")

    with tab3:
        st.markdown("### Lager Layout")
        # Vis Layout med hindringer og enveiskjøring
        fig_map = go.Figure()
        
        # Reoler (Hindringer)
        reoler = layout_df[layout_df["Type"] == "Reol"]
        fig_map.add_trace(go.Scatter(x=reoler["X"], y=reoler["Y"], mode='markers', marker=dict(symbol='square', size=15, color='black'), name='Reol (Sperret)'))
        
        # Plukkplasser
        plukk = layout_df[layout_df["Type"] == "Plukk"]
        # Fargelegg etter om de har artikkel eller ikke
        fig_map.add_trace(go.Scatter(x=plukk["X"], y=plukk["Y"], mode='markers', marker=dict(color='blue', size=10), name='Plukkplass', text=plukk["ArtikkelID"]))
        
        # Spesielle
        start = layout_df[layout_df["Type"] == "Start"]
        printer = layout_df[layout_df["Type"] == "Printer"]
        fig_map.add_trace(go.Scatter(x=start["X"], y=start["Y"], mode='markers+text', text="START", marker=dict(color='green', size=20), name='Start'))
        fig_map.add_trace(go.Scatter(x=printer["X"], y=printer["Y"], mode='markers+text', text="PRINT", marker=dict(color='red', size=20), name='Printer'))
        
        # Enveiskjøring piler (Enkelt utvalg)
        arrows = layout_df[layout_df["Retning"].notna()]
        if not arrows.empty:
            fig_map.add_trace(go.Scatter(x=arrows["X"], y=arrows["Y"], mode='text', text=arrows["Retning"], textfont=dict(color='orange', size=14), name='Enveiskjøring'))

        fig_map.update_layout(height=600, width=800, title="Oversiktskart (Svart = Reol/Hindring)")
        st.plotly_chart(fig_map)

    with tab4:
        st.dataframe(layout_df)
        st.dataframe(ordre_df)

else:
    st.info("Trykk på 'Generer Demo Data' i menyen til venstre for å starte.")
