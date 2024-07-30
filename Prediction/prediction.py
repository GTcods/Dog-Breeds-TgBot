import numpy as np
from tensorflow.keras.preprocessing import image


def preprocess_image(img, target_size=(224, 224)):
    # Convert the image to a numpy array
    img = img.resize(target_size)  # Resize to match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0

    return img_array


def predict_breed(model, img_array):
    # Predict the class probabilities
    predictions = model.predict(img_array)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    predicted_breed = class_names[predicted_class_index]
    return predicted_breed


class_names = ['Affenpinscher', 'Afghan Hound', 'Airedale Terrier', 'Akita', 'Alaskan Malamute', 'American Bulldog',
               'American Cocker Spaniel', 'American Eskimo Dog', 'American Foxhound', 'American Hairless Terrier',
               'American Pit Bull Terrier', 'American Staffordshire Terrier', 'American Water Spaniel',
               'Anatolian Shepherd Dog', 'Australian Cattle Dog', 'Australian Shepherd', 'Australian Terrier',
               'Azawakh', 'Basenji', 'Basset Hound', 'Beagle', 'Bearded Collie', 'Bloodhound', 'Border Terrier',
               'Borzoi', 'Boston Terrier', 'Bouvier des Flandres', 'Boxer', 'Boykin Spaniel', 'Briard', 'Brittany dog',
               'Brussels Griffon', 'Bull Terrier', 'Bulldog', 'Bullmastiff', 'Cairn Terrier', 'Canaan Dog',
               'Cane Corso', 'Cardigan Welsh Corgi', 'Cavalier King Charles Spaniel', 'Cesky Terrier',
               'Chesapeake Bay Retriever', 'Chihuahua', 'Chinese Crested', 'Chinese Shar-Pei', 'Chinook Dog',
               'Chow Chow', 'Clumber Spaniel', 'Cockapoo', 'Cocker Spaniel', 'Collie', 'Coonhound', 'Corgi',
               'Coton de Tulear', 'Curly-Coated Retriever', 'Dachshund', 'Dalmatian', 'Dandie Dinmont Terrier',
               'Doberman Pinscher', 'Dogue de Bordeaux', 'Dutch Shepherd', 'English Bulldog', 'English Cocker Spaniel',
               'English Foxhound', 'English Setter', 'English Springer Spaniel', 'English Toy Spaniel',
               'Entlebucher Mountain Dog', 'Eskimo Dog', 'Field Spaniel', 'Finnish Lapphund', 'Finnish Spitz',
               'Flat-Coated Retriever', 'French Bulldog', 'German Pinscher', 'German Shepherd',
               'German Shorthaired Pointer', 'German Wirehaired Pointer', 'Giant Schnauzer', 'Glen of Imaal Terrier',
               'Golden Retriever', 'Goldendoodle', 'Gordon Setter', 'Great Dane', 'Great Pyrenees',
               'Greater Swiss Mountain Dog', 'Greyhound', 'Harrier Dog', 'Havanese', 'Ibizan Hound',
               'Icelandic Sheepdog', 'Irish Red and White Setter', 'Irish Setter', 'Irish Terrier',
               'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound', 'Jack Russell Terrier',
               'Japanese Chin', 'Japanese Spitz', 'Keeshond', 'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labradoodle',
               'Labrador Retriever', 'Lakeland Terrier', 'Leonberger', 'Lhasa Apso', 'Lowchen', 'Maltese',
               'Manchester Terrier', 'Mastiff', 'Miniature Bull Terrier', 'Miniature Pinscher', 'Miniature Schnauzer',
               'Mudi', 'Neapolitan Mastiff', 'Newfoundland', 'Norfolk Terrier', 'Norwegian Buhund',
               'Norwegian Elkhound', 'Norwegian Lundehund', 'Norwich Terrier', 'Nova Scotia Duck Tolling Retriever',
               'Old English Sheepdog', 'Otterhound', 'Papillon', 'Pekingese', 'Pembroke Welsh Corgi',
               'Petit Basset Griffon Vendeen', 'Pharaoh Hound', 'Plott', 'Pointer', 'Polish Lowland Sheepdog',
               'Pomeranian', 'Poodle', 'Portuguese Water Dog', 'Pug', 'Puli', 'Pumi', 'Siberian Husky']
