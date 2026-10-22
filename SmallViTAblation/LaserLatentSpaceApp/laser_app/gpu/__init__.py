"""Elementy OpenGL 4.3 zakładki „Lasery i koloidy” (pyqtgraph GLGraphicsItem + shadery GLSL 4.30 z buforami SSBO).

Cała historia (wszystkie checkpointy: położenia, kierunki, mapy) trafia na kartę RAZ, po wczytaniu. W klatce CPU podaje tylko
ułamkowy indeks czasu u_t i opcje; interpolację między checkpointami, kolory, rozmiary, punkty załamania na sferze i mapy na zdjęciu
liczą shadery. Jedyne dane wysyłane w trakcie odtwarzania to natężenia z_jc (600 × 768) pary sąsiednich checkpointów dla wiązek,
wymieniane, gdy oś czasu przejdzie do następnej pary.

- `PlanarMotion`: panele 1–3 filmu 2/16 (ruch globalny, studnie i odkształcenie lokalne, drgania wokół centrów ze śladami);
- `ReflectorSphere`: film 12 (768 stałych reflektorów, 600 wiązek każdy, jedno załamanie na sferze, cząsteczki p_j = P W₂ (z − z̄));
- `ImageMaps`: kafelki zdjęcia z mapą 30 × 20 (atencja 6 bloków, centra, drgania, natężenie reflektora) i ramkami.

Moduły: base (wspólne), sphere_gl, focal_field_gl, planar_gl, image_maps_gl, network_passage_gl, valley_gl, schematic_gl, trap_map_gl, projections_gl."""
