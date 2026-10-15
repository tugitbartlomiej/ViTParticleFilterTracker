"""Scena Blendera układu laser-koloid z filmu 12 (zasada Z14): uklad_laser_koloid_12.blend.

- 768 stałych reflektorów (obudowa + soczewka w kolorze kanału) na zewnętrznej sferze RR = 50, ustawionych przodem do środka.
- Szklana sfera R = 34: wiązka z reflektora c biegnie prosto do punktu E_jc na sferze, tam załamuje się raz,
  a w środku idzie prosto w kierunku d_c do cząsteczki j. Wszystkie wiązki jednego reflektora są w sferze równoległe.
- 600 cząsteczek (tokeny), narzędzie na błękitno. Reflektor pokazowy 436: wszystkie 600 wiązek i matryca 30 × 20 diod z tyłu.
- Uczenie: 24 kroki kluczowe (kroki 0–22 974) interpolowane w Geometry Nodes; czas uczenia = właściwość sceny „uczenie”
  (0 … 23, sterownik bez Pythona), więc w Blenderze można go przewijać ręcznie (Properties → Scene → Custom Properties).
- Ujęcia: reflektor 436 z boku (tylko jego wiązki) → matryca diod → wszystkie wiązki → uczenie z obrotem kamery → koniec.
  Kamery przełączają znaczniki osi czasu. Napisy są w osobnej scenie „Montaz” (Video Sequencer) nad wyrenderowanym filmem.
Dane: blender_12_dane.npz z blender_12_eksport.py.
Uruchomienie: F:/Instalki/blender.exe -b --factory-startup --python-exit-code 1 -P blender_12_scena.py -- [podglad] [anim] [montaz] [cycles]"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
HERE = Path(__file__).resolve().parent
D = np.load(HERE / 'blender_12_dane.npz')
R, RR, C1 = float(D['R']), float(D['RR']), int(D['C1'])
NKL = len(D['KROKI'])
RFIX, KOL, NARZ = D['RFIX'], D['KOL'], D['NARZ'].astype(bool)
# oś czasu (30 kl./s): reflektor 436 z boku, matryca diod, wszystkie wiązki, uczenie, koniec
F_JEDEN, F_MATR, F_WSZ, F_UCZ, KROK_KL, OGON = 1, 181, 271, 421, 30, 90
F_UCZ_K = F_UCZ + KROK_KL * (NKL - 1)
F_END = F_UCZ_K + OGON
print('dane:', {k: D[k].shape for k in D.files}, flush=True)

# ---------------- scena ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.name = 'Uklad'
sc['uczenie'] = float(NKL - 1)
sc.frame_start, sc.frame_end, sc.render.fps = 1, F_END, 30
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.engine = 'CYCLES' if 'cycles' in ARGS else 'BLENDER_EEVEE_NEXT'
sc.eevee.taa_render_samples = 24
cy = sc.cycles
cy.samples, cy.use_denoising = 48, True
cy.max_bounces, cy.transparent_max_bounces = 4, 96
cy.diffuse_bounces = cy.glossy_bounces = 2
cy.transmission_bounces, cy.volume_bounces = 2, 0
try:
    pref = bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type = 'OPTIX'
    pref.get_devices()
    for dv in pref.devices:
        dv.use = dv.type == 'OPTIX'
    cy.device = 'GPU'
except Exception as e:                                             # noqa: BLE001
    print('GPU niedostępne:', e)
sc.view_settings.view_transform = 'AgX'
sc.view_settings.look = 'AgX - Punchy'
sc.world = bpy.data.worlds.new('Tlo')
sc.world.use_nodes = True
sc.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.004, 0.006, 0.012, 1)
sc.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0


def kolekcja(nazwa):
    c = bpy.data.collections.new(nazwa); sc.collection.children.link(c); return c


K_UKL, K_WSZ, K_KAM = kolekcja('Uklad'), kolekcja('Wiazki_wszystkie'), kolekcja('Kamery')


def siatka(nazwa, V, E=(), F=(), kol=K_UKL):
    me = bpy.data.meshes.new(nazwa)
    me.vertices.add(len(V)); me.vertices.foreach_set('co', np.asarray(V, 'f4').ravel())
    if len(E):
        me.edges.add(len(E)); me.edges.foreach_set('vertices', np.asarray(E, 'i4').ravel())
    if len(F):
        F = np.asarray(F, 'i4'); n = F.shape[1]
        me.loops.add(F.size); me.loops.foreach_set('vertex_index', F.ravel())
        me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', np.arange(0, F.size, n, dtype='i4'))
        me.polygons.foreach_set('loop_total', np.full(len(F), n, 'i4'))
    me.update(calc_edges=bool(len(F)))
    ob = bpy.data.objects.new(nazwa, me); kol.objects.link(ob)
    return ob


def atrybut(ob, nazwa, X, typ):
    X = np.asarray(X, 'f4')
    a = ob.data.attributes.new(nazwa, typ, 'POINT')
    a.data.foreach_set({'FLOAT': 'value', 'FLOAT_VECTOR': 'vector', 'FLOAT_COLOR': 'color'}[typ], X.ravel())


def klucze(ob, nazwa, X, typ):
    """X (NKL, n[, 3]) → atrybuty nazwa_0 … nazwa_{NKL-1}."""
    for k in range(NKL):
        atrybut(ob, f'{nazwa}_{k}', X[k], typ)


# ---------------- materiały ----------------
def mat_swiecacy(nazwa, kolor_attr=None, kolor=(1, 1, 1), sila_attr=None, sila=1.0, przezroczysty=True, rampa=None, mnoz_attr=None):
    """Emisja (addytywnie z przezroczystością): kolor z atrybutu lub stały, siła = sila · atrybut [· drugi atrybut]."""
    m = bpy.data.materials.new(nazwa); m.use_nodes = True
    nt = m.node_tree; N, L = nt.nodes, nt.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); em = N.new('ShaderNodeEmission')
    if kolor_attr:
        a = N.new('ShaderNodeAttribute'); a.attribute_name = kolor_attr
        L.new(a.outputs['Color'], em.inputs['Color'])
    elif rampa:
        a = N.new('ShaderNodeAttribute'); a.attribute_name = rampa
        cr = N.new('ShaderNodeValToRGB'); el = cr.color_ramp.elements
        stopy = [(0.0, (0.27, 0.0, 0.33)), (0.25, (0.23, 0.32, 0.55)), (0.5, (0.13, 0.57, 0.55)), (0.75, (0.37, 0.79, 0.38)), (1.0, (0.99, 0.91, 0.14))]
        el[0].position, el[0].color = stopy[0][0], (*stopy[0][1], 1); el[1].position, el[1].color = stopy[-1][0], (*stopy[-1][1], 1)
        for p, c in stopy[1:-1]:
            e = el.new(p); e.color = (*c, 1)
        L.new(a.outputs['Fac'], cr.inputs['Fac']); L.new(cr.outputs['Color'], em.inputs['Color'])
    else:
        em.inputs['Color'].default_value = (*kolor, 1)
    if sila_attr:
        a = N.new('ShaderNodeAttribute'); a.attribute_name = sila_attr
        mm = N.new('ShaderNodeMath'); mm.operation = 'MULTIPLY'; mm.inputs[1].default_value = sila
        L.new(a.outputs['Fac'], mm.inputs[0]); wyj = mm.outputs[0]
        if mnoz_attr:
            b = N.new('ShaderNodeAttribute'); b.attribute_name = mnoz_attr
            m2 = N.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'
            L.new(wyj, m2.inputs[0]); L.new(b.outputs['Fac'], m2.inputs[1]); wyj = m2.outputs[0]
        L.new(wyj, em.inputs['Strength'])
    else:
        em.inputs['Strength'].default_value = sila
    if przezroczysty:
        tr = N.new('ShaderNodeBsdfTransparent'); ad = N.new('ShaderNodeAddShader')
        L.new(tr.outputs[0], ad.inputs[0]); L.new(em.outputs[0], ad.inputs[1]); L.new(ad.outputs[0], out.inputs['Surface'])
        if hasattr(m, 'surface_render_method'):
            m.surface_render_method = 'BLENDED'
    else:
        L.new(em.outputs[0], out.inputs['Surface'])
    return m


def mat_szklo():
    """Sfera: przezroczysta z delikatnie świecącą krawędzią (bez drugiego załamania obrazu — załamanie wiązek jest w geometrii)."""
    m = bpy.data.materials.new('Szklo_sfera'); m.use_nodes = True
    nt = m.node_tree; N, L = nt.nodes, nt.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); lw = N.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.5
    pw = N.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 3.0
    L.new(lw.outputs['Facing'], pw.inputs[0])
    em = N.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.35, 0.65, 1.0, 1)
    ms = N.new('ShaderNodeMath'); ms.operation = 'MULTIPLY'; ms.inputs[1].default_value = 0.8
    L.new(pw.outputs[0], ms.inputs[0]); L.new(ms.outputs[0], em.inputs['Strength'])
    tr = N.new('ShaderNodeBsdfTransparent'); ad = N.new('ShaderNodeAddShader')
    L.new(tr.outputs[0], ad.inputs[0]); L.new(em.outputs[0], ad.inputs[1]); L.new(ad.outputs[0], out.inputs['Surface'])
    if hasattr(m, 'surface_render_method'):
        m.surface_render_method = 'BLENDED'
    return m


def mat_metal(nazwa, kolor=(0.08, 0.09, 0.1), szorstkosc=0.35):
    m = bpy.data.materials.new(nazwa); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*kolor, 1); b.inputs['Metallic'].default_value = 0.9; b.inputs['Roughness'].default_value = szorstkosc
    return m


M_WIAZKA = mat_swiecacy('Wiazki', kolor_attr='kol', sila_attr='jasnosc', sila=0.3, mnoz_attr='profil')
M_WIAZKA_C1 = mat_swiecacy('Wiazki_reflektora_436', kolor_attr='kol', sila_attr='jasnosc', sila=2.5, mnoz_attr='profil')
M_SOCZ = mat_swiecacy('Soczewki', kolor_attr='kol', sila_attr='moc', sila=4.0, przezroczysty=False)
M_OBUD = mat_metal('Obudowa', (0.03, 0.035, 0.04), 0.5)
M_CZ = mat_swiecacy('Czasteczki', kolor=(0.85, 1.0, 0.9), sila_attr='jasn', sila=3.0, przezroczysty=False)
M_NARZ = mat_swiecacy('Narzedzie', kolor=(0.29, 0.94, 1.0), sila=8.0, przezroczysty=False)
M_DIODY = mat_swiecacy('Diody_436', rampa='jasnosc', sila=6.0, przezroczysty=False)
M_SIATKA = mat_swiecacy('Siatka_sfery', kolor=(0.3, 0.55, 0.75), sila=0.5)


# ---------------- Geometry Nodes: interpolacja kroków kluczowych ----------------
class Drzewo:
    def __init__(self, nazwa):
        self.ng = bpy.data.node_groups.new(nazwa, 'GeometryNodeTree')
        self.ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        self.ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        self.N, self.L = self.ng.nodes, self.ng.links
        self.wej = self.N.new('NodeGroupInput'); self.wyj = self.N.new('NodeGroupOutput')
        czas = self.N.new('ShaderNodeValue'); czas.label = 'uczenie (krok kluczowy 0 … 23)'
        drv = czas.outputs[0].driver_add('default_value').driver; drv.type = 'AVERAGE'
        var = drv.variables.new(); var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'SCENE'; var.targets[0].id = sc; var.targets[0].data_path = '["uczenie"]'
        t = self.mat('MINIMUM', self.mat('MAXIMUM', czas.outputs[0], 0.0), float(NKL - 1))
        self.k0 = self.mat('FLOOR', t)
        self.k1 = self.mat('MINIMUM', self.mat('ADD', self.k0, 1.0), float(NKL - 1))
        self.u = self.mat('SUBTRACT', t, self.k0)

    def n(self, typ, **we):
        nd = self.N.new(typ)
        for k, v in we.items():
            s = nd.inputs[k]
            if isinstance(v, bpy.types.NodeSocket):
                self.L.new(v, s)
            else:
                s.default_value = v
        return nd

    def mat(self, op, a, b=None):
        nd = self.N.new('ShaderNodeMath'); nd.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                self.L.new(v, nd.inputs[i])
            else:
                nd.inputs[i].default_value = v
        return nd.outputs[0]

    def interp(self, nazwa, wektor):
        """wartość atrybutu nazwa_k interpolowana liniowo między krokami kluczowymi."""
        sw = []
        for idx in (self.k0, self.k1):
            s = self.N.new('GeometryNodeIndexSwitch'); s.data_type = 'VECTOR' if wektor else 'FLOAT'
            while len(s.index_switch_items) < NKL:
                s.index_switch_items.new()
            self.L.new(idx, s.inputs['Index']); sw.append(s)
        for k in range(NKL):
            na = self.N.new('GeometryNodeInputNamedAttribute'); na.data_type = 'FLOAT_VECTOR' if wektor else 'FLOAT'
            na.inputs['Name'].default_value = f'{nazwa}_{k}'
            for s in sw:
                self.L.new(na.outputs['Attribute'], s.inputs[k + 1])
        a, b = sw[0].outputs[0], sw[1].outputs[0]
        if wektor:
            d = self.N.new('ShaderNodeVectorMath'); d.operation = 'SUBTRACT'; self.L.new(b, d.inputs[0]); self.L.new(a, d.inputs[1])
            s_ = self.N.new('ShaderNodeVectorMath'); s_.operation = 'SCALE'; self.L.new(d.outputs[0], s_.inputs[0]); self.L.new(self.u, s_.inputs['Scale'])
            p = self.N.new('ShaderNodeVectorMath'); p.operation = 'ADD'; self.L.new(a, p.inputs[0]); self.L.new(s_.outputs[0], p.inputs[1])
            return p.outputs[0]
        return self.mat('ADD', a, self.mat('MULTIPLY', self.mat('SUBTRACT', b, a), self.u))

    def zapisz(self, geo, nazwa, wartosc, wektor=False):
        st = self.N.new('GeometryNodeStoreNamedAttribute'); st.data_type = 'FLOAT_VECTOR' if wektor else 'FLOAT'; st.domain = 'POINT'
        st.inputs['Name'].default_value = nazwa
        self.L.new(geo, st.inputs['Geometry']); self.L.new(wartosc, st.inputs['Value'])
        return st.outputs[0]

    def ustaw_pozycje(self, geo, poz):
        sp = self.n('GeometryNodeSetPosition', Geometry=geo, Position=poz); return sp.outputs[0]

    def material(self, geo, m):
        sm = self.n('GeometryNodeSetMaterial', Geometry=geo); sm.inputs['Material'].default_value = m; return sm.outputs[0]

    def koniec(self, geo, ob):
        self.L.new(geo, self.wyj.inputs[0])
        mod = ob.modifiers.new('Uczenie', 'NODES'); mod.node_group = self.ng
        return mod


def rury(dr, geo, promien, m, prog=None):
    """krawędzie → krzywe → rurki o przekroju 4-kątnym; z progiem: wiązki słabsze niż prog znikają, grubość rośnie z natężeniem."""
    if prog is not None:
        na = dr.n('GeometryNodeInputNamedAttribute', Name='jasnosc'); na.data_type = 'FLOAT'
        dl = dr.n('GeometryNodeDeleteGeometry', Geometry=geo, Selection=dr.mat('LESS_THAN', na.outputs['Attribute'], prog))
        geo = dl.outputs[0]
    mc = dr.n('GeometryNodeMeshToCurve', Mesh=geo)
    krz = mc.outputs[0]
    if prog is not None:
        na = dr.n('GeometryNodeInputNamedAttribute', Name='jasnosc'); na.data_type = 'FLOAT'
        r_ = dr.mat('MINIMUM', dr.mat('ADD', dr.mat('MULTIPLY', na.outputs['Attribute'], 0.6), 0.4), 1.6)
        krz = dr.n('GeometryNodeSetCurveRadius', Curve=krz, Radius=r_).outputs[0]
    ci = dr.n('GeometryNodeCurvePrimitiveCircle', Resolution=4, Radius=promien)
    cm = dr.n('GeometryNodeCurveToMesh', Curve=krz, **{'Profile Curve': ci.outputs['Curve']})
    return dr.material(cm.outputs[0], m)


# ---------------- wiązki ----------------
def wiazki(nazwa, BC, E, P, I, m, promien, kol=K_UKL):
    """4 wierzchołki na wiązkę: reflektor → E (na zewnątrz), E → cząsteczka (w sferze, prosto w kierunku d_c).
    E (NKL, n, 3) i P (NKL, n, 3) zależą od kroku; I (NKL, n) = jasność z_jc / 2."""
    n = len(BC)
    X = np.zeros((NKL, n, 4, 3), 'f4')
    X[:, :, 0] = RFIX[BC][None]; X[:, :, 1] = E; X[:, :, 2] = E; X[:, :, 3] = P
    X = X.reshape(NKL, n * 4, 3)
    E_ = np.stack([np.arange(n) * 4, np.arange(n) * 4 + 1, np.arange(n) * 4 + 2, np.arange(n) * 4 + 3], 1).reshape(-1, 2)
    ob = siatka(nazwa, X[-1], E_, kol=kol)
    klucze(ob, 'P', X, 'FLOAT_VECTOR')
    klucze(ob, 'I', np.repeat(I, 4, axis=1), 'FLOAT')
    atrybut(ob, 'kol', np.repeat(np.concatenate([0.25 + 0.75 * KOL[BC], np.ones((n, 1))], 1), 4, axis=0), 'FLOAT_COLOR')
    # profil jasności wzdłuż wiązki jak w filmie: na zewnątrz 0,35·(0,6 … 1), w sferze rośnie do cząsteczki
    atrybut(ob, 'profil', np.tile([0.21, 0.35, 0.15, 0.6], n), 'FLOAT')
    dr = Drzewo(f'GN_{nazwa}')
    g = dr.ustaw_pozycje(dr.wej.outputs[0], dr.interp('P', True))
    g = dr.zapisz(g, 'jasnosc', dr.interp('I', False))
    dr.koniec(rury(dr, g, promien, m, prog=0.05), ob)
    return ob


BC, BJ = D['BC'], D['BJ']
POS = D['POS']
wiazki('Wiazki_768_reflektorow', BC, D['BE'], POS[:, BJ], D['BI'], M_WIAZKA, 0.035, K_WSZ)
wiazki('Wiazki_reflektora_436', np.full(600, C1), D['HE'], POS, D['HI'], M_WIAZKA_C1, 0.045)

# ---------------- cząsteczki ----------------
ob = siatka('Czasteczki_600', POS[-1])
klucze(ob, 'P', POS, 'FLOAT_VECTOR'); klucze(ob, 'J', D['SJN'], 'FLOAT')
atrybut(ob, 'narz', NARZ.astype('f4'), 'FLOAT')
dr = Drzewo('GN_czasteczki')
g = dr.ustaw_pozycje(dr.wej.outputs[0], dr.interp('P', True))
g = dr.zapisz(g, 'jasn', dr.mat('ADD', dr.mat('MULTIPLY', dr.interp('J', False), 0.9), 0.1))
narz = dr.n('GeometryNodeInputNamedAttribute', Name='narz'); narz.data_type = 'FLOAT'
sel_n = dr.mat('GREATER_THAN', narz.outputs['Attribute'], 0.5)
ico = dr.n('GeometryNodeMeshIcoSphere', Radius=0.42, Subdivisions=2)
ring = dr.n('GeometryNodeMeshIcoSphere', Radius=0.75, Subdivisions=2)
ip = dr.n('GeometryNodeInstanceOnPoints', Points=g, Instance=ico.outputs['Mesh'])
ipn = dr.n('GeometryNodeInstanceOnPoints', Points=g, Selection=sel_n, Instance=ring.outputs['Mesh'])
r1 = dr.material(dr.n('GeometryNodeRealizeInstances', Geometry=ip.outputs[0]).outputs[0], M_CZ)
r2 = dr.material(dr.n('GeometryNodeRealizeInstances', Geometry=ipn.outputs[0]).outputs[0], M_NARZ)
jn = dr.N.new('GeometryNodeJoinGeometry'); dr.L.new(r2, jn.inputs[0]); dr.L.new(r1, jn.inputs[0])
mod = dr.koniec(jn.outputs[0], ob)

# ---------------- reflektory ----------------
ob = siatka('Reflektory_768', RFIX)
klucze(ob, 'M', D['MOCN'], 'FLOAT')
atrybut(ob, 'kol', np.concatenate([0.3 + 0.7 * KOL, np.ones((768, 1))], 1), 'FLOAT_COLOR')
dr = Drzewo('GN_reflektory')
g = dr.zapisz(dr.wej.outputs[0], 'moc', dr.mat('MULTIPLY', dr.interp('M', False), 1.0))
poz = dr.N.new('GeometryNodeInputPosition')
dk = dr.N.new('ShaderNodeVectorMath'); dk.operation = 'SCALE'; dr.L.new(poz.outputs[0], dk.inputs[0]); dk.inputs['Scale'].default_value = -1.0
al = dr.N.new('FunctionNodeAlignRotationToVector'); al.axis = 'Z'; dr.L.new(dk.outputs[0], al.inputs['Vector'])
obud = dr.material(dr.n('GeometryNodeMeshCube', Size=(1.2, 1.2, 2.0)).outputs['Mesh'], M_OBUD)
socz = dr.n('GeometryNodeMeshCylinder', Vertices=24, Radius=0.45, Depth=0.12)
socz_t = dr.n('GeometryNodeTransform', Geometry=socz.outputs['Mesh'], Translation=(0, 0, 1.04))
socz_m = dr.material(socz_t.outputs[0], M_SOCZ)
jn = dr.N.new('GeometryNodeJoinGeometry'); dr.L.new(socz_m, jn.inputs[0]); dr.L.new(obud, jn.inputs[0])
ip = dr.n('GeometryNodeInstanceOnPoints', Points=g, Instance=jn.outputs[0], Rotation=al.outputs[0])
dr.koniec(dr.n('GeometryNodeRealizeInstances', Geometry=ip.outputs[0]).outputs[0], ob)

# ---------------- matryca 30 × 20 diod reflektora 436 (z tyłu obudowy, na zewnątrz) ----------------
os_z = Vector((-RFIX[C1]).tolist()).normalized()                   # przód reflektora patrzy do środka
os_x = os_z.cross(Vector((0, 0, 1))).normalized(); os_y = os_z.cross(os_x)
krok, bok = 0.036, 0.03
kk, ww = np.meshgrid(np.arange(30), np.arange(20))
cx = (kk.ravel() - 14.5) * krok; cyy = (9.5 - ww.ravel()) * krok
rog = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * bok / 2
lok = np.stack([cx[:, None] + rog[None, :, 0], cyy[:, None] + rog[None, :, 1]], 2).reshape(-1, 2)
srod = Vector(RFIX[C1].tolist()) - os_z * 1.03                     # tylna ściana obudowy (od strony zewnętrznej)
Vd = np.array([srod - os_z * 0.01 + os_x * float(a) * -1 + os_y * float(b) for a, b in lok], 'f4')
ob = siatka('Matryca_diod_436', Vd, F=np.arange(2400).reshape(600, 4))
klucze(ob, 'Z', np.repeat(D['ZC1N'], 4, axis=1), 'FLOAT')
dr = Drzewo('GN_diody')
g = dr.zapisz(dr.wej.outputs[0], 'jasnosc', dr.interp('Z', False))
dr.koniec(dr.material(g, M_DIODY), ob)
tlo = siatka('Matryca_tlo_436', [srod + os_x * sx * 0.58 + os_y * sy * 0.4 - os_z * 0.005 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
             F=[[0, 1, 2, 3]])
tlo.data.materials.append(mat_metal('Matryca_plytka', (0.02, 0.02, 0.025), 0.6))

# ---------------- szklana sfera i jej siatka ----------------
bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=R)
sf = bpy.context.active_object; sf.name = 'Szklana_sfera_R34'; bpy.ops.object.shade_smooth()
sf.data.materials.append(mat_szklo())
for c in sf.users_collection:
    c.objects.unlink(sf)
K_UKL.objects.link(sf)
seg = []
for lon in np.radians(np.arange(0, 180, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p = R * np.stack([np.cos(t_) * np.cos(lon), np.cos(t_) * np.sin(lon), np.sin(t_)], 1); seg.append(p)
for lat in np.radians(np.arange(-72, 90, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p = R * np.stack([np.cos(t_) * np.cos(lat), np.sin(t_) * np.cos(lat), np.full_like(t_, np.sin(lat))], 1); seg.append(p)
V_ = np.concatenate(seg); E_ = np.concatenate([np.stack([np.arange(120), np.arange(1, 121)], 1) + 121 * i for i in range(len(seg))])
ob = siatka('Siatka_sfery', V_ * 1.002, E_)
dr = Drzewo('GN_siatka'); dr.koniec(rury(dr, dr.wej.outputs[0], 0.05, M_SIATKA), ob)

# ---------------- kamery ----------------
def cel(nazwa, p):
    e = bpy.data.objects.new(nazwa, None); e.location = p; K_KAM.objects.link(e); return e


def kamera(nazwa, p, cel_ob, ogniskowa=28.0):
    cd = bpy.data.cameras.new(nazwa); cd.lens = ogniskowa; cd.clip_end = 1000
    cam = bpy.data.objects.new(nazwa, cd); cam.location = p; K_KAM.objects.link(cam)
    tr = cam.constraints.new('TRACK_TO'); tr.target = cel_ob; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
    return cam


bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
srodek = cel('Cel_srodek', (0, 0, 0))
obrot = cel('Obrot_kamery', (0, 0, 0))
k_orb = kamera('Kamera_orbita', (175 * math.cos(math.radians(22)), 0, 175 * math.sin(math.radians(22))), srodek)
k_orb.parent = obrot
obrot.rotation_euler = (0, 0, 0); obrot.keyframe_insert('rotation_euler', index=2, frame=F_WSZ)
obrot.rotation_euler = (0, 0, math.radians(300)); obrot.keyframe_insert('rotation_euler', index=2, frame=F_END)
# z boku względem kierunku d_c reflektora 436: wiązki w sferze widać jako równoległe kreski
d1 = Vector(D['DIRS'][-1][C1].tolist()).normalized()
bok_ = d1.cross(Vector((0, 0, 1))).normalized()
k_bok = kamera('Kamera_reflektor_436_z_boku', bok_ * 150 + Vector((0, 0, 12)), srodek)
cel_m = cel('Cel_matryca_436', srod + os_z * 1.5)
k_mat = kamera('Kamera_matryca_436', srod - os_z * 5.5 + os_x * 2.0 + os_y * 1.6, cel_m, 35.0)
for f, kam in ((F_JEDEN, k_bok), (F_MATR, k_mat), (F_WSZ, k_orb)):
    mk = sc.timeline_markers.new(kam.name, frame=f); mk.camera = kam
sc.camera = k_orb

# czas uczenia: koniec (krok 22 974) w ujęciach pokazowych, potem od kroku 0 do końca
for f, v in ((1, NKL - 1), (F_UCZ - 1, NKL - 1), (F_UCZ, 0), (F_UCZ_K, NKL - 1), (F_END, NKL - 1)):
    sc['uczenie'] = float(v); sc.keyframe_insert('["uczenie"]', frame=f)
sc['uczenie'] = float(NKL - 1)
# na początku tylko wiązki reflektora 436
wsz = bpy.data.objects['Wiazki_768_reflektorow']
for f, h in ((1, True), (F_WSZ, False)):
    wsz.hide_render = h; wsz.keyframe_insert('hide_render', frame=f)
    wsz.hide_viewport = h; wsz.keyframe_insert('hide_viewport', frame=f)
wsz.hide_render = wsz.hide_viewport = False

# ---------------- kompozycja: poświata ----------------
sc.use_nodes = True
ct = sc.node_tree
rl = ct.nodes.get('Render Layers') or ct.nodes.new('CompositorNodeRLayers')
co = ct.nodes.get('Composite') or ct.nodes.new('CompositorNodeComposite')
gl = ct.nodes.new('CompositorNodeGlare'); gl.glare_type = 'FOG_GLOW'
for at, v in (('threshold', 0.8), ('size', 7), ('mix', 0.25), ('quality', 'HIGH')):
    if hasattr(gl, at):
        setattr(gl, at, v)
ct.links.new(rl.outputs['Image'], gl.inputs['Image']); ct.links.new(gl.outputs['Image'], co.inputs['Image'])

# ---------------- montaż: film z Blendera + napisy (Video Sequencer) ----------------
SUROWY = HERE / 'blender_12_surowy.mp4'
mont = bpy.data.scenes.new('Montaz')
mont.render.resolution_x, mont.render.resolution_y, mont.render.fps = 1920, 1080, 30
mont.frame_start, mont.frame_end = 1, F_END
se = mont.sequence_editor_create()
PASY = se.strips if hasattr(se, 'strips') else se.sequences
if SUROWY.exists():
    PASY.new_movie('film', str(SUROWY), channel=1, frame_start=1)
pl = lambda x, n=2: f'{x:.{n}f}'.replace('.', ',')


def napis(tekst, f0, f1, y, rozm, kolor=(0.92, 0.95, 1.0), kanal=2):
    t = PASY.new_effect(tekst[:30], 'TEXT', channel=kanal, frame_start=f0, frame_end=f1)
    t.text, t.font_size, t.location, t.color = tekst, rozm, (0.5, y), (*kolor, 1.0)
    for a, v in (('use_shadow', True), ('shadow_color', (0, 0, 0, 0.9)), ('anchor_x', 'CENTER'), ('anchor_y', 'CENTER'),
                 ('align_x', 'CENTER'), ('align_y', 'CENTER'), ('use_box', True), ('box_color', (0.02, 0.03, 0.04, 0.55)), ('box_margin', 0.008)):
        if hasattr(t, a):
            setattr(t, a, v)
    return t


BIA, POM, NIE = (0.92, 0.95, 1.0), (1.0, 0.72, 0.45), (0.6, 0.85, 1.0)
UJ = [(F_JEDEN, F_MATR, f'Reflektor {C1}: 600 wiązek, jedno załamanie na sferze, w środku równolegle',
       ['Galwo kieruje wiązkę j w punkt sfery, z którego prosta w kierunku d_c trafia w cząsteczkę j.',
        'Sfera załamuje wiązkę raz; w środku wszystkie wiązki reflektora biegną równolegle (wspólne wagi v_c), jasność = z_jc.']),
      (F_MATR, F_WSZ, f'Matryca 30 × 20 diod reflektora {C1}',
       ['Jedna dioda na token obrazu (siatka 30 × 20); jasność diody j = aktywacja z_jc neuronu c dla tokenu j.']),
      (F_WSZ, F_UCZ, '768 stałych reflektorów, szklana sfera R = 34, 600 cząsteczek',
       ['Reflektory stoją na spirali Fibonacciego; kanał c stoi tam, gdzie leży najbliżej za swoim kierunkiem pchnięcia (przypisanie węgierskie).',
        'Na każdą cząsteczkę działa 768 wektorów siły (po jednym z każdego reflektora).  Pokazano 48 najsilniejszych wiązek na reflektor.']),
      (F_UCZ, F_UCZ_K, 'Uczenie: reflektory stoją, punkty załamania wędrują po sferze',
       ['Kierunek d_c zmienia się raz dla wszystkich 600 wiązek reflektora, więc sfera załamuje je pod innym kątem;  galwa przestawiają cel.',
        'Tempo: pierwsze 12 kroków kluczowych to epoki 0–2, następne 11 to epoki 2–21.']),
      (F_UCZ_K, F_END + 1, 'Po 21 epokach',
       [f'Pary reflektorów o przeciwnych kierunkach (cos < −0,5): {D["PARY"][0]:.0%} → {D["PARY"][-1]:.0%}.   IoU {pl(float(D["IOU"][-1]))}.',
        'Szybko: jasności wiązek i celowanie galw;  powoli (uczenie): kierunki d_c, czyli punkty i kąty załamania na sferze.'])]
for f0, f1, tyt, pod in UJ:
    napis(tyt, f0, f1, 0.95, 40, BIA, 2)
    for n, s_ in enumerate(pod):
        napis(s_, f0, f1, 0.895 - 0.04 * n, 24, POM, 3 + n)
napis('Dane: wagi fc2 bloku 6 z 341 checkpointów i aktywacje 768 neuronów zdjęcia TRAIN 0.  Optyka poglądowa; d_c = P v̂_c, P = 3 główne składowe.',
      1, F_END + 1, 0.03, 18, NIE, 6)
for k in range(NKL):                                                # licznik epok w czasie uczenia
    f0 = F_UCZ + KROK_KL * k - KROK_KL // 2 if k else F_UCZ
    f1 = min(F_UCZ + KROK_KL * k + KROK_KL // 2, F_UCZ_K) if k < NKL - 1 else F_END + 1
    napis(f'epoka {pl(float(D["EPOKI"][k]))}   krok {int(D["KROKI"][k]):,}'.replace(',', ' ') + f'   IoU {pl(float(D["IOU"][k]))}'
          + f'   pary przeciwne {D["PARY"][k]:.0%}', f0, f1, 0.08, 26, POM, 7)
mont.render.image_settings.file_format = 'FFMPEG'
mont.render.ffmpeg.format, mont.render.ffmpeg.codec, mont.render.ffmpeg.constant_rate_factor = 'MPEG4', 'H264', 'HIGH'
mont.render.filepath = str(HERE / 'blender_12_uklad.mp4')

sc.frame_set(F_UCZ_K)
plik = HERE / 'uklad_laser_koloid_12.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(plik), compress=True)
print('zapisano', plik, flush=True)

# ---------------- podgląd i render ----------------
if 'podglad' in ARGS:
    sc.render.image_settings.file_format = 'PNG'
    for nazwa, klatka in (('1_reflektor_436_z_boku', 120), ('2_matryca_436', 230), ('3_wszystkie', 300),
                          ('4_uczenie_poczatek', F_UCZ + 60), ('5_koniec', F_UCZ_K + 30)):
        sc.frame_set(klatka)
        sc.render.filepath = str(HERE / f'blender_12_{nazwa}.png')
        bpy.ops.render.render(write_still=True, scene=sc.name)
        print('podgląd', nazwa, sc.camera.name, flush=True)
if 'anim' in ARGS:
    sc.render.image_settings.file_format = 'FFMPEG'
    sc.render.ffmpeg.format, sc.render.ffmpeg.codec, sc.render.ffmpeg.constant_rate_factor = 'MPEG4', 'H264', 'HIGH'
    sc.render.filepath = str(SUROWY)
    bpy.ops.render.render(animation=True, scene=sc.name)
if 'montaz' in ARGS:
    if SUROWY.exists() and not any(p_.type == 'MOVIE' for p_ in PASY):
        PASY.new_movie('film', str(SUROWY), channel=1, frame_start=1)
    bpy.ops.render.render(animation=True, scene=mont.name)
