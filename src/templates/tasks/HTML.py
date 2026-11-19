HTML_PROMPT = """[FORMAT:html] Transcris le document en fragment HTML structuré dans un bloc ```html, sans explication. Pas de doctype, html, head ou body, ni de CSS.

Paragraphes et titres : conserve leur sémantique, utilise un <span coord="..."> par ligne physique, sans fusionner les lignes. Conserve aside, signature, page_number, del et br. Tableaux : table, thead/tbody/tfoot, tr, th/td, cellules vides, rowspan/colspan. Conserve les relations utiles id et data-target.

Coordonnées dans l'attribut coord="xmin,ymin,xmax,ymax,angle", sans espaces ni crochets. Coordonnées normalisées [0,1000], angle en degrés. Jamais de coordonnées dans le texte, ni de data-coords. Exemple :
```html
<p><span coord="52,38,680,61,0">Le sieur Dupont vous écrit</span></p>
```

Respecte le texte, les espaces de transcription et l'ordre de lecture. Utilise seulement l'apostrophe droite ' dans le texte. Pas d'attributs de présentation class/style/lang/data-page."""
