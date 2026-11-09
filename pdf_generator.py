from fpdf import FPDF

# Redondea para mostrar (evita 0.20000000000000004 y -0).
def n(x):
    return f"{round(x, 4) + 0.0:g}"


def generar_pdf(p):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    #  Titulo
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Reporte - Perceptron Monocapa (Compuerta AND)",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    #  Parametros 
    primero = p.pasos[0]
    ultima = p.epocas[-1]
    convergio = "Si" if ultima["errores"] == 0 else "No"

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Parametros", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Pesos iniciales: w1 = {n(primero['w1_ant'])}, "
                   f"w2 = {n(primero['w2_ant'])}, b = {n(primero['b_ant'])}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Tasa de aprendizaje (eta): {n(p.eta)}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Epocas ejecutadas: {len(p.epocas)}  |  Convergio: {convergio}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Funcion escalon: f(z) = 1 si z >= 0, de lo contrario f(z) = 0",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    #  Tabla de epocas 
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Tabla resumen por epocas", new_x="LMARGIN", new_y="NEXT")

    anchos = [25, 35, 35, 35, 35]
    encabezados = ["Epoca", "w1", "w2", "b", "Errores"]

    pdf.set_font("Helvetica", "B", 11)
    for ancho, texto in zip(anchos, encabezados):
        pdf.cell(ancho, 8, texto, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 11)
    for e in p.epocas:
        fila = [e["epoca"], n(e["w1"]), n(e["w2"]), n(e["b"]), e["errores"]]
        for ancho, valor in zip(anchos, fila):
            pdf.cell(ancho, 8, str(valor), border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    #  Procedimiento 
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Procedimiento paso a paso", new_x="LMARGIN", new_y="NEXT")

    for e in p.epocas:
        if pdf.get_y() > 240:          # evita titulo de época solo al final
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"Epoca {e['epoca']} (errores: {e['errores']})",
                 new_x="LMARGIN", new_y="NEXT")

        pasos_epoca = [s for s in p.pasos if s["epoca"] == e["epoca"]]
        for i, s in enumerate(pasos_epoca, start=1):
            if pdf.get_y() > 250:      # evita partir una muestra entre páginas
                pdf.add_page()
            lineas = [
                f"Muestra {i}: (x1, x2) = ({s['x1']}, {s['x2']}), t = {s['t']}",
                f"  z = ({s['x1']} * {n(s['w1_ant'])}) + "
                f"({s['x2']} * {n(s['w2_ant'])}) + ({n(s['b_ant'])}) = {n(s['z'])}",
                f"  y = f(z) = {s['y']}   e = t - y = {s['t']} - {s['y']} = {s['e']}",
                f"  w1 = {n(s['w1_ant'])} + {n(p.eta)} * ({s['e']}) * {s['x1']} "
                f"= {n(s['w1_nuevo'])}",
                f"  w2 = {n(s['w2_ant'])} + {n(p.eta)} * ({s['e']}) * {s['x2']} "
                f"= {n(s['w2_nuevo'])}",
                f"  b = {n(s['b_ant'])} + {n(p.eta)} * ({s['e']}) * 1 "
                f"= {n(s['b_nuevo'])}",
            ]
            pdf.set_font("Helvetica", "", 10)
            for linea in lineas:
                pdf.multi_cell(0, 5, linea, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        pdf.ln(2)

    return bytes(pdf.output())