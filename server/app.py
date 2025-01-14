from flask import request, url_for, redirect, jsonify
from flask_cors import CORS
from PIL import Image
import os
from langchain_community.llms import Ollama
from database import create_app
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from routes import init_routes
from utils.helpers import get_ingredients_from_image
from model_rag import load_model

rag_chain, retriever = load_model()

load_dotenv()

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

init_routes(app)

bcrypt = Bcrypt(app)

ollama = Ollama(
    base_url='http://localhost:11434',
    model="gemma2:2b"
)

# Ruta principal de la app
@app.route('/')
def index():
    return "App activo"

@app.route('/ingredientes_detectados', methods=['POST'])
def detectar_ingredientes():
    try:
        # Verificar si hay texto en la solicitud
        text = request.form.get('text', '')
        
        # Verificar si hay una imagen en la solicitud
        images = request.files.getlist('images')

        all_ingredients = []
        
        print(f"Texto recibido: {text}")
        print(f"Cantidad de imagenes recibidas: {len(images)}")

        if images:
            for image_file in images:
                image = Image.open(image_file)
                ingredients = get_ingredients_from_image(image)
                print(f"Ingredientes predichos: {ingredients}")

                if not ingredients:
                    return jsonify({'error': 'No se detectaron ingredientes en una o más imágenes.'}), 200

                all_ingredients.extend(ingredients)
            
            # Eliminar duplicados
            all_ingredients = list(set(all_ingredients))

        if text:
            # Si hay texto, añadirlo a los ingredientes detectados
            user_ingredients = text.split(',')
            all_ingredients.extend([ingredient.strip() for ingredient in user_ingredients])

            # Eliminar duplicados después de agregar los ingredientes del texto
            all_ingredients = list(set(all_ingredients))

        if all_ingredients:
            return jsonify({'response': all_ingredients})
        else:
            return jsonify({'error': 'No se recibieron ingredientes ni en texto ni en imágenes.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/consulta_ollama', methods=['POST'])
def consulta_ollama():
    try:
        ingredients = request.json.get('ingredients', [])
        use_rag = request.json.get('use_rag', False)

        if ingredients:
            all_ingredients = list(set(ingredient.strip() for ingredient in ingredients))

            print("INGREDIENTES:", all_ingredients)

            if use_rag:
                generated_text = ""
                for chunk in rag_chain.stream({"context": retriever, "question": all_ingredients}):
                    generated_text += chunk.content
            else:
                prompt = f"Dame una receta sencilla con los siguientes ingredientes: {', '.join(all_ingredients)}. Evita incluir elementos no relacionados o creativos. Quiero que me dividas la respuesta en 4 categorias (Titulo, Ingredientes, Preparación y Consejos) donde cada una de ellas tienen que comenzar con las siguientes exactas palabras segun corresponda a cada una de ellas: 'Titulo', 'Ingredientes:', 'Peparación:' y 'Consejos'."
                generated_text = ollama.invoke(prompt)

            return jsonify({'response': generated_text})
        else:
            return jsonify({'error': 'No se recibieron ingredientes ni en texto ni en imágenes.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)})

# Si el usuario quiere ingresar a cualquier pagina que no este
# definida, se lo redirecciona al inicio
def pagina_no_encotrada(error):
    # return render_template('404.html'), 404 (opcion para mostrar un index personalizado en vez de solo redireccionar)
    return redirect(url_for('index'))

if __name__ == "__main__":  
    # Manejamos los errores con el metodo que creamos
    app.register_error_handler(404, pagina_no_encotrada)
    app.run(host='0.0.0.0', port=5000,debug = True)
    CORS(app, origins=[os.getenv("URL_REACT")])
