import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64



# Load the TFlite model for core/non-core classification model
core_non_core_model_path = 'core_nonecore_model_20240330.tflite'
core_non_core_interpreter = tf.lite.Interpreter(model_path=core_non_core_model_path)
core_non_core_interpreter.allocate_tensors()


# Define core_non_core_classes dictionary
core_non_core_classes = {
    1: "none",
    0: "core"
}
# Load the TFLite model
tflite_model = 'finetuning_field_lithology_vgg_model_20240330.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()

# Set page title and favicon
st.set_page_config(page_title="RockAI", page_icon="🪨", layout="wide")

# Custom CSS for styling the buttons
button_style = """
    <style>
        .button-bar {
            background-color: #0000E6;
            padding: 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .button-bar button {
            background-color: #0000E6;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            transition: background-color 0.3s;
        }
        .button-bar button:hover {
            background-color: #0000E6;
        }
    </style>
"""

# Display the blue color bar with buttons
st.markdown(button_style, unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="button-bar">
        <button onclick="window.location.href='home_url_here'">Home</button>
        <button onclick="window.location.href='https://www.slb.com'">About Us</button>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """<a href="https://www.slb.com">
    <img src="data:image/jpg;base64,{}" width="150">
    </a>""".format(
        base64.b64encode(open("slb.jpg", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)

st.image("logo_rockai.PNG", width=250)


# Write welcome message
def show_welcome_message():
    st.write(
        """
        # Mining Lithology Predictor

        Welcome to RockAI! 👋 The 'RockAI' is an innovative application designed to enhance mining operations across various environments. This innovative tool utilizes advanced deep learning techniques, offering an accessible and straightforward solution for Mining operations in various environments (rock types of prediction from core images). The state-of-the-art tool is not limited to subsurface mining but extends its capabilities to various mining types, offering a broader range of geological exploration and mineral prospecting. ✨
        """
    )

# Button to display welcome message
if st.button("Welcome"):
    show_welcome_message()

# Option buttons for uploading image or taking picture
option = st.radio("Choose Option", ("Upload Image", "Take Picture"))

# Placeholder for class descriptions (you can fill this from the internet)
class_descriptions = {
    'AMPH': ["Amphibolite", "Amphibolite is a metamorphic rock that contains plagioclase feldspar and amphibole minerals, usually hornblende or actinolite. It is typically dark-colored and dense, with a weakly foliated or flaky structure. It can form from the metamorphism of mafic igneous rocks such as basalt and gabbro, or from the metamorphism of clay-rich sedimentary rocks such as marl or graywacke (Class 0)"],
    'BREC': ["Breccia", "Breccia is a type of sedimentary rock that consists of large angular fragments of other rocks or minerals, cemented together by a fine-grained material. The fragments are usually gravel-sized or larger and have different shapes and colors depending on the source material. Breccia can form from various processes such as weathering, erosion, volcanic activity, or impact events (Class 1)"],
    'BSLT': ["Basalt", "Basalt is an igneous rock formed from the rapid cooling of low-viscosity lava rich in magnesium and iron. It is an aphanitic (fine-grained) extrusive rock that often appears dark in color (Class 2)"],
    'DACT': ["Dacite", "Dacite is a fine-grained igneous rock that is normally light in color. It is often porphyritic. Dacite is found in lava flows, lava domes, dikes, sills, and pyroclastic debris. It is a rock type usually found on continental crust above subduction zones, where a relatively young oceanic plate has melted below (Class 3)"],
    'DOLR': ["Dolerite", "Dolerite, also known as diabase, is an intriguing igneous rock formed from molten magma that cools and solidifies beneath the Earth’s surface. Its composition includes plagioclase feldspar (appearing as white to light gray crystals) and pyroxene minerals (primarily augite, which contributes to its dark color). Minor minerals like olivine, magnetite, and apatite may also be present. Dolerite’s slow crystallization process results in a fine-grained texture and durability. It forms deep within the Earth’s crust or mantle and often occurs as slabs and blocks. The name “dolerite” comes from Greek words meaning “poison stone,” reflecting its dark color and toxic nature. (Class 4)"],
    'GNSS': ["Gneiss", "Gneiss pronounced as 'nais' is a common and widely distributed type of metamorphic rock. It forms through high-temperature and high-pressure metamorphic processes acting on formations composed of igneous or sedimentary rocks. (Class 5)"],
    'GRNT': ["Granite", "Granite is a light-colored igneous rock with large grains that can be seen with the eye1. It forms from the slow cooling of magma below the Earths surface. It is mainly composed of quartz and feldspar, with some other minerals. (Class 6)"],
    'QTZT': ["Quartzite", "Quartzite is a nonfoliated metamorphic rock composed almost entirely of quartz. It forms when a quartz-rich sandstone is altered by the heat, pressure, and chemical activity of metamorphism. Metamorphism recrystallizes the sand grains and the silica cement that binds them together. The result is a network of interlocking quartz grains of incredible strength. (Class 7)"],
    'SCHT': ["Schist", "Schist is a metamorphic rock that has thin, flat mineral grains arranged in layers or bands. It is mainly composed of platy minerals like mica, feldspar, and quartz. It is easy to split into thin plates or flakes. (Class 8)"],
    'SDST': ["Sandstone", "Sandstone is a sedimentary rock made from sand-sized grains of minerals or fragments of other rocks. It can have different compositions and textures depending on the type and size of the grains. Most sandstone is composed of quartz or feldspar (Class 9)"],
    'SHLE': ["Shale", "Shale is a fine-grained sedimentary rock that forms from the compaction of mud consisting of clay and tiny particles of minerals and organic compounds. Shale is the most common sedimentary rock in the Earths crust and it has a property called fissility, which means it can split into thin layers. (Class 10)"],
    'SLST': ["Siltstone", "Siltstone is a sedimentary rock composed mainly of silt-sized particles. It forms where water, wind, or ice deposit silt, and the silt is then compacted and cemented into a rock. Silt accumulates in sedimentary basins throughout the world. It represents a level of current, wave, or wind energy between where sand and mud accumulate. These include fluvial, aeolian, tidal, coastal, lacustrine, deltaic, glacial, paludal, and shelf environments. Sedimentary structures such as layering, cross-bedding, ripple marks, erosional contacts, and fossils provide evidence of these environments. Siltstone is much less common than sandstone and shale. The rock units are usually thinner and less extensive. Only rarely is one notable enough to merit a stratigraphic name. (Class 11)"]
}

# Placeholder for class information
class_information = {
    'AMPH': "https://geology.com/rocks/amphibolite.shtml",
    'BREC': "https://geology.com/rocks/breccia.shtml",
    'BSLT': "https://geology.com/rocks/bslt.shtml",
    'DACT': "https://geology.com/rocks/dacite.shtml",
    'DOLR': "https://geology.com/rocks/dolerite.shtml",
    'GNSS': "https://geology.com/rocks/gneiss.shtml",
    'GRNT': "https://geology.com/rocks/granite.shtml",
    'QTZT': "https://geology.com/rocks/quartzite.shtml",
    'SCHT': "https://geology.com/rocks/schist.shtml",
    'SDST': "https://geology.com/rocks/sandstone.shtml",
    'SHLE': "https://geology.com/rocks/shale.shtml",
    'SLST': "https://geology.com/rocks/siltstone.shtml",
}


# Define the descriptions and options for each test
tests_info = {
    'Hardness Test': {
        'description': 'Use a steel knife, glass, or a standard hardness kit to scratch the rock.',
        'options': {
            'Cannot scratch with a knife (soft)': 'soft',
            'Can scratch with a knife but not glass (medium)': 'medium',
            'Can scratch glass (hard)': 'hard'
        }
    },
    'Acid Test': {
        'description': 'Apply a few drops of dilute hydrochloric acid to the rock.',
        'options': {
            'No reaction (non-carbonate)': 'non-carbonate',
            'Fizzes or reacts (carbonate)': 'carbonate'
        }
    },
    'Magnetism Test': {
        'description': 'Hold a magnet close to the rock.',
        'options': {
            'Not magnetic': 'not magnetic',
            'Magnetic': 'magnetic'
        }
    },
    'Foliation Test': {
        'description': 'Examine the rock for layers or a banded appearance.',
        'options': {
            'Not foliated': 'not foliated',
            'Foliated': 'foliated'
        }
    }
}

# Placeholder for physical tests information
physical_tests_info = {
    'Dacite': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Generally not magnetic.",
        'Foliation': "Not foliated."
    },
    'Dolerite': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "May be magnetic due to iron-rich minerals.",
        'Foliation': "Not foliated."
    },
    'Granite': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Generally not magnetic.",
        'Foliation': "Not foliated."
    },
    'Basalt': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "May be magnetic due to iron-rich minerals.",
        'Foliation': "Not foliated."
    },
    'Sandstone': {
        'Hardness': "Can scratch with a knife but not glass (medium).",
        'Acid Test': "No reaction (non-carbonate) for quartz sandstones, fizzes for calcareous sandstones.",
        'Magnetism': "Not magnetic.",
        'Foliation': "Not foliated."
    },
    'Shale': {
        'Hardness': "Cannot scratch with a knife (soft).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Not magnetic.",
        'Foliation': "Not foliated."
    },
    'Siltstone': {
        'Hardness': "Can scratch with a knife but not glass (medium).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Not magnetic.",
        'Foliation': "Not foliated."
    },
    'Breccia': {
        'Hardness': "Varies depending on the cement and clasts, typically can scratch with a knife but not glass (medium).",
        'Acid Test': "Fizzes or reacts if the cement or clasts are carbonate.",
        'Magnetism': "Not magnetic, unless it contains magnetic minerals.",
        'Foliation': "Not foliated."
    },
    'Gneiss': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate), unless marble bands are present.",
        'Magnetism': "Not magnetic.",
        'Foliation': "Foliated (banded appearance)."
    },
    'Amphibolite': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Not magnetic, generally, but can contain magnetic minerals.",
        'Foliation': "May be foliated depending on its formation."
    },
    'Schist': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Not magnetic.",
        'Foliation': "Foliated (visible layers or sheets)."
    },
    'Quartzite': {
        'Hardness': "Can scratch glass (hard).",
        'Acid Test': "No reaction (non-carbonate).",
        'Magnetism': "Not magnetic.",
        'Foliation': "Not foliated, but may show evidence of its original bedding."
    }
}

# Placeholder for further investigation information
further_investigation_info = {
    'Dacite': "Further investigation: Dacite may contain crystals of quartz, feldspar, and biotite or hornblende. Look for these minerals under a magnifying glass or microscope.",
    'Dolerite': "Further investigation: Examine the rock under a microscope for the presence of plagioclase feldspar, pyroxene, and olivine crystals.",
    'Granite': "Further investigation: Granite often contains visible crystals of quartz, feldspar, and mica. Look for these minerals and examine their shapes and colors.",
    'Basalt': "Further investigation: Basalt typically contains small crystals of plagioclase feldspar and pyroxene. Examine the rock under a magnifying glass or microscope to identify these minerals.",
    'Sandstone': "Further investigation: Examine the sandstone for evidence of its depositional environment, such as cross-bedding, ripple marks, or fossilized remains.",
    'Shale': "Further investigation: Shale may contain fossilized remains of plants or animals. Examine the rock under a magnifying glass or microscope to look for these features.",
    'Siltstone': "Further investigation: Siltstone often contains layers or lenses of different colors or grain sizes. Examine the rock closely to identify these variations.",
    'Breccia': "Further investigation: Examine the breccia for evidence of its origin, such as angular clasts embedded in a finer-grained matrix. Look for clues about the processes that formed the rock.",
    'Gneiss': "Further investigation: Gneiss may contain bands or layers of different minerals. Examine the rock closely to identify these variations and their relationships to each other.",
    'Amphibolite': "Further investigation: Amphibolite may contain crystals of amphibole minerals such as hornblende or biotite. Look for these minerals under a magnifying glass or microscope.",
    'Schist': "Further investigation: Schist may contain visible layers or sheets of mica, quartz, or other minerals. Examine the rock closely to identify these features and their orientations.",
    'Quartzite': "Further investigation: Quartzite may contain remnants of its original sand grains or evidence of metamorphic recrystallization. Examine the rock under a magnifying glass or microscope to look for these features."
}


# Function to predict lithology
def predict_lithology(image):
    # Resize the image
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Convert the NumPy array to a compatible data type (uint8)
    image_array = image_array.astype(np.uint8)
    # Process the image for prediction
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    class_names = {'AMPH': 0, 'BREC': 1, 'BSLT': 2, 'DACT': 3, 'DOLR': 4, 'GNSS': 5, 'GRNT': 6, 'QTZT': 7, 'SCHT': 8, 'SDST': 9, 'SHLE': 10, 'SLST': 11}
    predicted_class_index = np.argmax(predictions[0])
    predicted_label_name = list(class_names.keys())[predicted_class_index]
    # Extract the predicted label name and description
    predicted_label_info = class_descriptions[predicted_label_name]
    # Return predicted_label_name, predicted_label_info for further processing
    return predicted_label_name, predicted_label_info, image


def preprocess_image(image):
    # Resize the image
    image = image.resize((224, 224))
    # Convert the image to a NumPy array and change data type to float32
    image_array = np.array(image).astype(np.float32)
    # Normalize the image data
    image_array = image_array / 255.0
    # Expand dimensions to match the input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def classify_core_non_core(tflite_interpreter, preprocessed_image):
    # Set input tensor
    input_details = tflite_interpreter.get_input_details()
    tflite_interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    # Run inference
    tflite_interpreter.invoke()
    # Get the output tensor
    output_details = tflite_interpreter.get_output_details()
    output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
    # Interpret the output
    predicted_label_name = 'core' if output_data[0] < 0.5 else 'none'
    return predicted_label_name


if option == "Upload Image":
    # Upload Image Part
    uploaded_image = st.file_uploader("Choose a borehole core image for Mining lithology prediction...", type=["tif", "tiff", "jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        # Classify the image as core/non-core
        predicted_label_name = classify_core_non_core(core_non_core_interpreter, preprocessed_image)

        if predicted_label_name == "core":
            # Call the predict_lithology function
            predicted_label_name, predicted_label_info, image = predict_lithology(image)
            # Display lithology prediction results
            # Extract the predicted label name and description
            predicted_label_info = class_descriptions[predicted_label_name]

            # Create a two-column layout
            col1, col2 = st.columns(2)

            # Display the predicted label above the image in bold and larger text
            predicted_label_full_name = class_descriptions[predicted_label_name][0]
            col1.write(f"**Predicted Lithology: [{predicted_label_full_name}]({class_information[predicted_label_name]})**", unsafe_allow_html=True)

            # Display the image in the first column
            col1.image(image, caption=f"Predicted Lithology: {predicted_label_full_name}", width=400)

            # Display the class description in the second column
            col2.markdown(f"#### Description for {predicted_label_full_name}:")
            col2.write(predicted_label_info[1])
            col2.markdown(f"For further information, [click here]({class_information[predicted_label_name]})")


            # Display the physical tests and further investigation information based on the predicted rock type
            if predicted_label_full_name in physical_tests_info:
                col2.markdown("### Physical Tests:")
                for test, description in physical_tests_info[predicted_label_full_name].items():
                    col2.write(f"**{test}:** {description}")

                col2.markdown("### Further Investigation:")
                col2.write(further_investigation_info[predicted_label_full_name])

            # Button for displaying physical tests description
            if st.button("Description of Physical Tests"):
                # Print all tests along with their descriptions using Streamlit's write function
                for test_name, test_info in tests_info.items():
                    st.write(f"## {test_name}:")
                    st.write(f"**Description:** {test_info['description']}")
                    st.write("Options for User to Select:")
                    for option_desc, option_value in test_info['options'].items():
                        st.write(f"- {option_desc}")
                    st.write()
        else:
            st.markdown("**Please focus on the CORE section by adjusting the angle, zoom, and lighting**")

elif option == "Take Picture":
    # Take picture Part
    take_picture = st.camera_input("Take Picture")

    if take_picture is not None:
        # Decode the image data from the UploadedFile object
        image = Image.open(io.BytesIO(take_picture.read()))

        # Convert the image to grayscale if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        # Classify the image as core/non-core
        predicted_label_name = classify_core_non_core(core_non_core_interpreter, preprocessed_image)

        if predicted_label_name == "core":
            # Call the predict_lithology function
            predicted_label_name, predicted_label_info, image = predict_lithology(image)
            # Display lithology prediction results
            # Extract the predicted label name and description
            predicted_label_info = class_descriptions[predicted_label_name]

            # Create a two-column layout
            col1, col2 = st.columns(2)

            # Display the predicted label above the image in bold and larger text
            predicted_label_full_name = class_descriptions[predicted_label_name][0]
            col1.write(f"**Predicted Lithology: [{predicted_label_full_name}]({class_information[predicted_label_name]})**", unsafe_allow_html=True)

            # Display the image in the first column
            col1.image(image, caption=f"Predicted Lithology: {predicted_label_full_name}", width=400)

            # Display the class description in the second column
            col2.markdown(f"#### Description for {predicted_label_full_name}:")
            col2.write(predicted_label_info[1])
            col2.markdown(f"For further information, [click here]({class_information[predicted_label_name]})")

            # Display the further investigation information
            col2.markdown("### Further Investigation:")
            col2.write(further_investigation_info[predicted_label_full_name])

            # Display the physical tests and further investigation information based on the predicted rock type
            if predicted_label_full_name in physical_tests_info:
                col2.markdown("### Physical Tests:")
                for test, description in physical_tests_info[predicted_label_full_name].items():
                    col2.write(f"**{test}:** {description}")

                col2.markdown("### Further Investigation:")
                col2.write(further_investigation_info[predicted_label_full_name])

            # Button for displaying physical tests description
            if st.button("Description of Physical Tests"):
                # Print all tests along with their descriptions using Streamlit's write function
                for test_name, test_info in tests_info.items():
                    st.write(f"## {test_name}:")
                    st.write(f"**Description:** {test_info['description']}")
                    st.write("Options for User to Select:")
                    for option_desc, option_value in test_info['options'].items():
                        st.write(f"- {option_desc}")
                    st.write()
        else:
            st.markdown("**Please focus on the CORE section by adjusting the angle, zoom, and lighting**")
