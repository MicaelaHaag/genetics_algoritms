"""
Algoritmo genético - Evolución de una frase
Materia: Técnicas, Entornos y Aplicaciones de Inteligencia Artificial (UNRN)

PROBLEMA
    Descubrir una FRASE OBJETIVO (por defecto "HOLA MUNDO") partiendo de palabras
    escritas al azar.
        Individuo : una palabra (ej: "QXLA MPNDO")
        Genes     : cada letra de la palabra
        Fitness   : cantidad de letras que están en la posición correcta
        Cruza     : el hijo toma el inicio de un padre y el final del otro
        Mutación  : a veces una letra cambia por otra al azar

QUÉ MUESTRA LA VENTANA
    Tres poblaciones IDÉNTICAS al inicio (misma semilla), cada una evolucionando con
    un método de selección distinto, para poder compararlos:
        Ruleta  : una ruleta real; el sector de cada individuo es proporcional a su fitness.
        Torneo  : grupos de k individuos que compiten; gana el de mayor fitness.
        Ranking : tabla de posiciones; la probabilidad depende del puesto, no del fitness.

CICLO DE UNA GENERACIÓN
    evaluar fitness -> SELECCIÓN -> cruza -> mutación -> elitismo -> nueva población

Requisitos: solo Python 3 con Tkinter (en Ubuntu: sudo apt install python3-tk).
"""

import bisect
import math
import random
import tkinter as tk
import unicodedata
from tkinter import font as tkfont

# --------------------------------------------------------------------------
# Parámetros generales
# --------------------------------------------------------------------------
ALFABETO = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # genes posibles (el espacio también)
FRASE_DEFECTO = "HOLA MUNDO"
SEED = 7                  # misma semilla -> misma población inicial para los 3 métodos
MAX_GEN = 600             # la ejecución se detiene sola en esta generación
EXPONENTE_EXTREMO = 3     # "escala extrema": se selecciona con fitness^3 en lugar de fitness

COLORES = {"Ruleta": "#d6453d", "Torneo": "#3a6ea5", "Ranking": "#3f9142"}
PALETA = ["#f4a6a0", "#f7c98b", "#f3e58a", "#b9e3a0", "#9edbc8", "#9fcdf0",
          "#b9b6f2", "#e2b0ee", "#f2a9cf", "#d6c4a8", "#c5d6dc", "#e8d3a2"]


# --------------------------------------------------------------------------
# 1. FITNESS
# --------------------------------------------------------------------------
def fitness(palabra, objetivo):
    """Cantidad de letras correctas (misma letra en la misma posición)."""
    return sum(1 for a, b in zip(palabra, objetivo) if a == b)


# --------------------------------------------------------------------------
# 2. MÉTODOS DE SELECCIÓN  (teoría expuesta en clase)
#    Cada función devuelve los índices de los n padres elegidos.
# --------------------------------------------------------------------------
def seleccion_ruleta(fit, n, rng):
    """
    RULETA (fitness-proportionate selection).
    La probabilidad de elegir al individuo i es proporcional a su fitness:

                  f_i
        P(i) = ---------
                Σ f_j

    Se arma la ruleta con las probabilidades acumuladas (cada individuo ocupa un
    sector) y se 'gira' n veces: cada giro es un número aleatorio r en [0, 1) que
    cae en el sector de algún individuo.

    Limitaciones (ver informe):
      - Depende de la escala: un individuo muy superior domina la ruleta.
      - La versión estándar NO funciona con fitness negativos. Una salida habitual
        es desplazar todos los valores sumando |mínimo|.
    """
    if min(fit) < 0:                                   # tratamiento de fitness negativos
        minimo = min(fit)
        fit = [f - minimo for f in fit]
    total = sum(fit)
    if total == 0:                                     # nadie tiene fitness: todos iguales
        prob = [1 / len(fit)] * len(fit)
    else:
        prob = [f / total for f in fit]                # P(i) = f_i / Σ f_j
    acumulado, s = [], 0.0
    for p in prob:                                     # sectores de la ruleta
        s += p
        acumulado.append(s)
    giros = [rng.random() for _ in range(n)]           # n giros
    idxs = [min(bisect.bisect_right(acumulado, r), len(fit) - 1) for r in giros]
    return idxs, prob, giros


def seleccion_torneo(fit, n, k, rng):
    """
    TORNEO (tournament selection).
    Para cada padre: se sortean k individuos al azar y gana el de mejor fitness.
    No se calcula ninguna probabilidad global: solo se COMPARA, por lo que no
    depende de la escala del fitness. El tamaño k controla la presión de selección:
        k = 2 -> presión baja | k = 3 -> intermedia | k = N -> siempre gana el mejor.
    """
    k = min(k, len(fit))
    idxs, torneos = [], []
    for _ in range(n):
        competidores = rng.sample(range(len(fit)), k)
        ganador = max(competidores, key=lambda i: fit[i])
        idxs.append(ganador)
        torneos.append((competidores, ganador))
    return idxs, torneos


def seleccion_ranking(fit, n, rng):
    """
    RANKING (rank selection) - variante: asignación proporcional simple al ranking.
    1) Se ordena la población por fitness (peor = rank 1, mejor = rank N).
    2) La probabilidad depende de la POSICIÓN, no del valor del fitness:

                  rank_i
        P(i) = -----------
                Σ rank_j

    Importa el orden, no la distancia entre fitness. Nota: "Rank Selection" no tiene
    una única fórmula (existen linear y exponential ranking); acá se usa la
    proporcional simple, igual que en el informe. Los empates se ordenan por
    posición en la lista.
    """
    N = len(fit)
    orden = sorted(range(N), key=lambda i: fit[i])     # de peor a mejor
    rank = [0] * N
    for posicion, i in enumerate(orden):
        rank[i] = posicion + 1                         # rank 1..N
    total = sum(rank)
    prob = [r / total for r in rank]                   # P(i) = rank_i / Σ rank_j
    idxs = rng.choices(range(N), weights=prob, k=n)    # se sortea con esas probabilidades
    return idxs, rank, prob


# --------------------------------------------------------------------------
# 3. CRUZA + MUTACIÓN + ELITISMO  (generación de la nueva población)
# --------------------------------------------------------------------------
def reproducir(pop, fit, padres_idx, n_elite, pm, rng):
    """
    - Elitismo : los n_elite mejores pasan directo a la nueva generación.
    - Cruza    : de un punto. hijo = inicio del padre 1 + final del padre 2.
    - Mutación : cada letra del hijo cambia por una al azar con probabilidad pm.
    """
    orden = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
    nueva = [pop[i] for i in orden[:n_elite]]                           # elitismo
    L = len(pop[0])
    for j in range(0, len(padres_idx), 2):
        p1, p2 = pop[padres_idx[j]], pop[padres_idx[j + 1]]
        corte = rng.randint(1, L - 1)
        hijo = p1[:corte] + p2[corte:]                                  # cruza
        hijo = "".join(rng.choice(ALFABETO) if rng.random() < pm else g  # mutación
                       for g in hijo)
        nueva.append(hijo)
    return nueva


def palabra_aleatoria(L, rng):
    return "".join(rng.choice(ALFABETO) for _ in range(L))


# --------------------------------------------------------------------------
# 4. UNA POBLACIÓN POR MÉTODO (lógica, sin interfaz)
# --------------------------------------------------------------------------
class Metodo:
    def __init__(self, nombre):
        self.nombre = nombre

    def reiniciar(self, pop0, rng, cfg, objetivo):
        self.objetivo = objetivo
        self.pop = list(pop0)
        self.rng = rng
        self.gen = 0
        self.mejor, self.div = [], []
        self.encontrado = None            # generación en la que apareció la frase completa
        self._registrar()
        self.seleccionar(cfg)

    def _registrar(self):
        self.fit = [fitness(p, self.objetivo) for p in self.pop]
        self.mejor.append(max(self.fit))
        self.div.append(len(set(self.pop)))          # diversidad = palabras distintas
        if self.encontrado is None and max(self.fit) == len(self.objetivo):
            self.encontrado = self.gen

    def seleccionar(self, cfg):
        """Elige los padres de la generación actual y arma los datos para dibujarla."""
        fit_sel = [f ** cfg["escala"] for f in self.fit]       # escala extrema: f^3
        N = len(self.pop)
        self.elite_sel = cfg["elite"]
        n = 2 * (N - cfg["elite"])                             # 2 padres por hijo
        f = self.fit

        if self.nombre == "Ruleta":
            idxs, prob, giros = seleccion_ruleta(fit_sel, n, self.rng)
            a, b = idxs[0], idxs[1]
            suf = f"^{cfg['escala']}" if cfg["escala"] != 1 else ""
            lineas = [
                f"P(i) = f_i / Σf    (Σf{suf} = {sum(fit_sel)})",
                f"Giro 1 -> #{a:02d} (f={f[a]}, P={prob[a]*100:.1f}%)",
                f"Giro 2 -> #{b:02d} (f={f[b]}, P={prob[b]*100:.1f}%)",
            ]
            self.vis = dict(tipo="ruleta", prob=prob, giro1=giros[0], ganador=a)
        elif self.nombre == "Torneo":
            idxs, torneos = seleccion_torneo(fit_sel, n, cfg["k"], self.rng)
            lineas = [
                f"Torneo k={cfg['k']}: gana el mejor del grupo (solo compara)",
                f"Torneo 1 -> gana #{torneos[0][1]:02d} (f={f[torneos[0][1]]})",
                f"Torneo 2 -> gana #{torneos[1][1]:02d} (f={f[torneos[1][1]]})",
            ]
            self.vis = dict(tipo="torneo", torneos=torneos[:2], k=cfg["k"])
        else:  # Ranking
            idxs, rank, prob = seleccion_ranking(fit_sel, n, self.rng)
            a, b = idxs[0], idxs[1]
            lineas = [
                f"P(i) = rank_i / Σrank    (Σrank = {sum(rank)})",
                f"Giro 1 -> #{a:02d} (rank {rank[a]}, f={f[a]})",
                f"Giro 2 -> #{b:02d} (rank {rank[b]}, f={f[b]})",
            ]
            self.vis = dict(tipo="ranking", rank=rank, prob=prob)

        self.padres_idx = idxs
        self.padres_unicos = set(idxs)
        lineas.append(f"Padres distintos elegidos: {len(self.padres_unicos)} de {N}")
        self.log = "\n".join(lineas)

    def avanzar(self, cfg):
        self.pop = reproducir(self.pop, self.fit, self.padres_idx, self.elite_sel,
                              cfg["pm"], self.rng)
        self.gen += 1
        self._registrar()
        self.seleccionar(cfg)


# --------------------------------------------------------------------------
# 5. INTERFAZ GRÁFICA (Tkinter)
# --------------------------------------------------------------------------
W, H = 460, 552                 # tamaño de cada columna (un método por columna)
Y_FILAS, ALTO_FILA = 66, 16     # lista de población
Y_VIS = 266                     # zona del dibujo propio de cada método
Y_LOG = 476                     # texto explicativo


def normalizar_frase(texto):
    """Mayúsculas, sin acentos y solo letras A-Z y espacio."""
    texto = unicodedata.normalize("NFD", texto.upper())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return "".join(c for c in texto if c in ALFABETO).strip()


class App:
    def __init__(self, root):
        self.root = root
        self.listo = False
        self.root.title("Algoritmo genético - Métodos de selección")
        self.root.resizable(False, False)
        self.metodos = [Metodo("Ruleta"), Metodo("Torneo"), Metodo("Ranking")]
        self.corriendo = False
        self.job = None
        self.reinicios = 0
        self.objetivo = FRASE_DEFECTO
        self.v_extremo = tk.BooleanVar(value=False)

        self.f_mono = tkfont.Font(family="Courier", size=11, weight="bold")
        self.f_mono_s = tkfont.Font(family="Courier", size=9)
        self.cw = self.f_mono.measure("M")
        self.cw_s = self.f_mono_s.measure("M")

        # --- barra superior
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(top, text="Algoritmo genético: evolución de una frase",
                 font=("Arial", 15, "bold")).pack(side="left")
        self.lbl_estado = tk.Label(top, text="", font=("Arial", 11))
        self.lbl_estado.pack(side="right")

        # --- controles (fila 1)
        ctl = tk.Frame(root)
        ctl.pack(fill="x", padx=8, pady=4)
        tk.Label(ctl, text="Frase objetivo:").pack(side="left")
        self.entry = tk.Entry(ctl, width=16, font=("Courier", 12, "bold"))
        self.entry.insert(0, FRASE_DEFECTO)
        self.entry.pack(side="left", padx=4)
        self.entry.bind("<Return>", lambda e: self.aplicar_frase())
        tk.Button(ctl, text="Aplicar", command=self.aplicar_frase).pack(side="left", padx=(0, 16))
        self.b_run = tk.Button(ctl, text="Iniciar", width=9, command=self.alternar)
        self.b_run.pack(side="left", padx=2)
        tk.Button(ctl, text="Paso", width=7, command=self.paso).pack(side="left", padx=2)
        tk.Button(ctl, text="Reiniciar", width=9, command=lambda: self.reiniciar()).pack(side="left", padx=2)
        tk.Checkbutton(ctl, text="Escala extrema (fitness³)", variable=self.v_extremo,
                       command=self.reseleccionar).pack(side="left", padx=16)

        # --- controles (fila 2: sliders)
        sc = tk.Frame(root)
        sc.pack(fill="x", padx=8)
        self.s_n = self._slider(sc, "Población", 6, 12, 12, 1)
        self.s_k = self._slider(sc, "k del torneo", 2, 6, 3, 1)
        self.s_pm = self._slider(sc, "Mutación por letra (%)", 0, 20, 8, 0.5)
        self.s_e = self._slider(sc, "Elitismo", 0, 3, 2, 1)
        self.s_vel = self._slider(sc, "Velocidad (ms por generación)", 50, 1500, 150, 50)
        self.s_n.bind("<ButtonRelease-1>", lambda e: self.reiniciar(nueva_semilla=False))
        self.s_k.config(command=lambda v: self.reseleccionar())
        self.s_e.config(command=lambda v: self.reseleccionar())

        # --- columnas: una por método
        cols = tk.Frame(root)
        cols.pack(padx=8, pady=(4, 0))
        self.cv = []
        for i in range(3):
            c = tk.Canvas(cols, width=W, height=H, bg="white",
                          highlightthickness=1, highlightbackground="#bbbbbb")
            c.grid(row=0, column=i, padx=2)
            self.cv.append(c)

        # --- gráficos de evolución
        self.cv_graf = tk.Canvas(root, width=3 * W + 12, height=170, bg="white",
                                 highlightthickness=1, highlightbackground="#bbbbbb")
        self.cv_graf.pack(padx=8, pady=6)

        self.listo = True
        self.reiniciar(nueva_semilla=False)

    def _slider(self, parent, texto, desde, hasta, valor, paso):
        s = tk.Scale(parent, from_=desde, to=hasta, resolution=paso, orient="horizontal",
                     label=texto, length=230)
        s.set(valor)
        s.pack(side="left", padx=4)
        return s

    # ---- configuración vigente (se lee en vivo de los controles)
    def cfg(self):
        N = int(self.s_n.get())
        return dict(k=int(self.s_k.get()), pm=float(self.s_pm.get()) / 100,
                    elite=min(int(self.s_e.get()), N - 1),
                    escala=EXPONENTE_EXTREMO if self.v_extremo.get() else 1)

    # ---- acciones
    def aplicar_frase(self):
        frase = normalizar_frase(self.entry.get())
        if not 3 <= len(frase) <= 14:
            self.lbl_estado.config(text="Frase inválida: 3 a 14 letras (A-Z y espacio)", fg="#b00020")
            return
        self.objetivo = frase
        self.entry.delete(0, "end")
        self.entry.insert(0, frase)
        self.reiniciar()

    def reiniciar(self, nueva_semilla=True):
        if not self.listo:
            return
        if nueva_semilla:
            self.reinicios += 1
        semilla = SEED + self.reinicios
        N, L = int(self.s_n.get()), len(self.objetivo)
        rng0 = random.Random(semilla)
        pop0 = [palabra_aleatoria(L, rng0) for _ in range(N)]      # población inicial COMÚN
        cfg = self.cfg()
        for m in self.metodos:
            m.reiniciar(pop0, random.Random(semilla + 1000), cfg, self.objetivo)
        self.pausar()
        self.refrescar()

    def reseleccionar(self):
        if not self.listo:
            return
        cfg = self.cfg()
        for m in self.metodos:
            m.seleccionar(cfg)
        self.refrescar()

    def alternar(self):
        self.pausar() if self.corriendo else self.iniciar()

    def iniciar(self):
        self.corriendo = True
        self.b_run.config(text="Pausar")
        self.tick()

    def pausar(self):
        self.corriendo = False
        self.b_run.config(text="Iniciar")
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None

    def paso(self):
        cfg = self.cfg()
        for m in self.metodos:
            m.avanzar(cfg)
        self.refrescar()

    def tick(self):
        self.job = None
        if not self.corriendo:
            return
        terminaron = all(m.encontrado is not None for m in self.metodos)
        if terminaron or self.metodos[0].gen >= MAX_GEN:
            self.pausar()
            self.refrescar()
            return
        self.paso()
        self.job = self.root.after(int(self.s_vel.get()), self.tick)

    # ---- dibujo
    def refrescar(self):
        if not self.listo:
            return
        m0 = self.metodos[0]
        todos = all(m.encontrado is not None for m in self.metodos)
        if todos:
            texto, color = "¡Los tres métodos encontraron la frase!", "#1b7a2e"
        else:
            texto, color = f"Generación {m0.gen}   |   objetivo: \"{self.objetivo}\"", "black"
        self.lbl_estado.config(text=texto, fg=color)
        for m, cv in zip(self.metodos, self.cv):
            self.dibujar_columna(m, cv)
        self.dibujar_graficos()

    def palabra(self, cv, x, y, texto, objetivo, font, cw):
        """Dibuja una palabra letra por letra: verde si es correcta, gris si no."""
        for i, (c, o) in enumerate(zip(texto, objetivo)):
            ok = c == o
            cv.create_text(x + i * cw, y, text=c if c != " " else "·", anchor="w", font=font,
                           fill="#1b9e3e" if ok else "#9a9a9a")

    def dibujar_columna(self, m, cv):
        cv.delete("all")
        color = COLORES[m.nombre]
        L, N = len(m.objetivo), len(m.pop)
        mejor = max(m.fit)

        # encabezado
        cv.create_text(10, 16, text=m.nombre, anchor="w", font=("Arial", 15, "bold"), fill=color)
        cv.create_text(W - 10, 16, text=f"mejor: {mejor}/{L}", anchor="e", font=("Arial", 12, "bold"))
        if m.encontrado is not None:
            cv.create_text(10, 38, anchor="w", font=("Arial", 10, "bold"), fill="#1b7a2e",
                           text=f"¡Frase encontrada en la generación {m.encontrado}!")
        else:
            cv.create_text(10, 38, anchor="w", font=("Arial", 10), fill="#666666",
                           text=f"Generación {m.gen}: buscando...")
        cv.create_text(10, 54, anchor="w", font=("Arial", 8), fill="#666666",
                       text="Población      ● elegido como padre     ★ mejor     verde = letra correcta")

        # población
        ibest = max(range(N), key=lambda i: m.fit[i])
        for i, (pal, f) in enumerate(zip(m.pop, m.fit)):
            y = Y_FILAS + 8 + i * ALTO_FILA
            if i in m.padres_unicos:
                cv.create_oval(6, y - 5, 16, y + 5, fill="#27c4d8", outline="#0a7f8f")
            if i == ibest:
                cv.create_text(22, y, text="★", font=("Arial", 11), fill="#e0a800")
            cv.create_text(38, y, text=f"#{i:02d}", anchor="w", font=self.f_mono_s, fill="#555555")
            self.palabra(cv, 74, y, pal, m.objetivo, self.f_mono, self.cw)
            xf = 74 + L * self.cw + 10
            cv.create_text(xf, y, text=f"{f}/{L}", anchor="w", font=self.f_mono_s)
            cv.create_rectangle(318, y - 4, 438, y + 4, outline="#dddddd")
            cv.create_rectangle(318, y - 4, 318 + 120 * f / L, y + 4, fill=color, outline=color)

        cv.create_line(8, Y_VIS - 6, W - 8, Y_VIS - 6, fill="#cccccc")
        v = m.vis
        if v["tipo"] == "ruleta":
            self.dibujar_ruleta(cv, m, v)
        elif v["tipo"] == "torneo":
            self.dibujar_torneo(cv, m, v)
        else:
            self.dibujar_ranking(cv, m, v)
        cv.create_text(10, Y_LOG, text=m.log, anchor="nw", font=self.f_mono_s, fill="#222222")

    # -- Ruleta: sectores proporcionales al fitness; el giro 1 cae bajo la flecha ▼
    def dibujar_ruleta(self, cv, m, v):
        cv.create_text(10, Y_VIS + 4, anchor="w", font=("Arial", 9, "bold"),
                       text="Ruleta: el sector de cada individuo es P(i) = f_i / Σf")
        cx, cy, R = 120, 376, 84
        giro = v["giro1"] * 360                       # grados (sentido horario desde arriba)
        acum = 0.0
        for i, p in enumerate(v["prob"]):
            if p <= 0:
                continue
            ini, fin = acum * 360, (acum + p) * 360   # sector [ini, fin] en horario
            acum += p
            extent = min(fin - ini, 359.99)
            ganador = i == v["ganador"]
            cv.create_arc(cx - R, cy - R, cx + R, cy + R, start=90 - (fin - giro), extent=extent,
                          fill=PALETA[i % len(PALETA)], outline="black" if ganador else "white",
                          width=3 if ganador else 1)
            if fin - ini >= 16:
                ang = math.radians((ini + fin) / 2 - giro)
                cv.create_text(cx + 0.66 * R * math.sin(ang), cy - 0.66 * R * math.cos(ang),
                               text=f"#{i:02d}", font=("Arial", 8, "bold"))
        cv.create_polygon(cx - 9, cy - R - 16, cx + 9, cy - R - 16, cx, cy - R + 2,
                          fill="#222222", outline="")
        notas = ["Cuanto más fitness,", "más grande el sector", "y más chances de salir.", "",
                 "▼ marca dónde cayó", "el giro 1 (se dibuja", "girada para que el", "ganador quede abajo",
                 "de la flecha)."]
        for j, t in enumerate(notas):
            cv.create_text(236, Y_VIS + 40 + j * 15, text=t, anchor="w", font=("Arial", 9), fill="#555555")

    # -- Torneo: grupos de k que compiten, gana el mejor
    def dibujar_torneo(self, cv, m, v):
        cv.create_text(10, Y_VIS + 4, anchor="w", font=("Arial", 9, "bold"),
                       text=f"Torneo: se sortean k = {v['k']} individuos y gana el de mayor fitness")
        y = Y_VIS + 20
        for t, (comp, ganador) in enumerate(v["torneos"], start=1):
            cv.create_text(10, y + 5, anchor="w", font=("Arial", 9, "bold"), fill=COLORES["Torneo"],
                           text=f"Torneo {t}: compiten {len(comp)} -> gana #{ganador:02d}")
            for j, i in enumerate(comp):
                col, fila = j % 3, j // 3
                x0, y0 = 10 + col * 148, y + 16 + fila * 34
                gan = i == ganador
                cv.create_rectangle(x0, y0, x0 + 142, y0 + 30,
                                    fill="#cdeccd" if gan else "#f3f3f3",
                                    outline="#2e8b3d" if gan else "#bbbbbb", width=2 if gan else 1)
                cv.create_text(x0 + 71, y0 + 9, text=m.pop[i], font=self.f_mono_s)
                cv.create_text(x0 + 71, y0 + 22, font=("Arial", 8, "bold" if gan else "normal"),
                               text=f"#{i:02d}  f={m.fit[i]}" + ("  GANA" if gan else ""))
            filas = (len(comp) - 1) // 3 + 1
            y += 22 + filas * 34 + 4

    # -- Ranking: tabla de posiciones, la probabilidad depende del puesto
    def dibujar_ranking(self, cv, m, v):
        cv.create_text(10, Y_VIS + 4, anchor="w", font=("Arial", 9, "bold"),
                       text="Ranking: se ordena y P(i) = rank_i / Σrank (importa el orden)")
        rank, prob = v["rank"], v["prob"]
        orden = sorted(range(len(rank)), key=lambda i: rank[i], reverse=True)   # mejor primero
        pmax = max(prob)
        for fila, i in enumerate(orden):
            y = Y_VIS + 26 + fila * 14
            if i in m.padres_unicos:
                cv.create_oval(6, y - 4, 14, y + 4, fill="#27c4d8", outline="#0a7f8f")
            cv.create_text(22, y, anchor="w", font=self.f_mono_s,
                           text=f"puesto {rank[i]:>2}   #{i:02d}   f={m.fit[i]}   P={prob[i]*100:4.1f}%")
            cv.create_rectangle(300, y - 4, 440, y + 4, outline="#dddddd")
            cv.create_rectangle(300, y - 4, 300 + 140 * prob[i] / pmax, y + 4,
                                fill=COLORES["Ranking"], outline=COLORES["Ranking"])

    # -- Gráficos de evolución (mejor fitness y diversidad)
    def dibujar_graficos(self):
        cv = self.cv_graf
        cv.delete("all")
        L, N = len(self.objetivo), len(self.metodos[0].pop)
        gen = max(30, self.metodos[0].gen)
        self.grafico(cv, 50, 28, 560, 110, "Mejor fitness (letras correctas) por generación",
                     [(COLORES[m.nombre], m.mejor, m.nombre) for m in self.metodos], L, gen)
        self.grafico(cv, 730, 28, 560, 110, "Diversidad (palabras distintas) por generación",
                     [(COLORES[m.nombre], m.div, m.nombre) for m in self.metodos], N, gen)

    def grafico(self, cv, x0, y0, w, h, titulo, series, ymax, xmax):
        cv.create_text(x0, y0 - 14, text=titulo, anchor="w", font=("Arial", 10, "bold"))
        cv.create_rectangle(x0, y0, x0 + w, y0 + h, outline="#999999")
        for frac in (0, 0.5, 1):
            y = y0 + h - frac * h
            cv.create_line(x0, y, x0 + w, y, fill="#eeeeee")
            cv.create_text(x0 - 6, y, text=f"{ymax * frac:g}", anchor="e", font=("Arial", 8))
        cv.create_text(x0, y0 + h + 10, text="0", font=("Arial", 8))
        cv.create_text(x0 + w, y0 + h + 10, text=f"gen {xmax}", font=("Arial", 8), anchor="e")
        for k, (color, ys, nombre) in enumerate(series):
            pts = [(x0 + g / xmax * w, y0 + h - v / ymax * h) for g, v in enumerate(ys)]
            if len(pts) > 1:
                cv.create_line(*[c for p in pts for c in p], fill=color, width=2)
            else:
                cv.create_oval(pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3, fill=color, outline=color)
            cv.create_text(x0 + w - 70, y0 + 12 + k * 14, text=f"— {nombre}", anchor="w",
                           font=("Arial", 8, "bold"), fill=color)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
