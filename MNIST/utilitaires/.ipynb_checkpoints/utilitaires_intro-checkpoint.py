from .utilitaires_mnist_2 import *
from .utilitaires_common import *


notebook_id = 4

start_analytics_session(notebook_id)

try:
    # For dev environment
    from strings_intro import *
except ModuleNotFoundError: 
    pass

def calculer_score_zone_custom(): 
    global e_train
    x = get_variable('x')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    caracteristique = get_variable('caracteristique')

    if not check_is_defined(x) or not check_is_defined(r_petite_caracteristique) or not check_is_defined(r_grande_caracteristique):
        return

    def algorithme(d):
        k = caracteristique(d)
        if k < x:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
    
    def cb(score):
        validation_score_zone_custom()

    calculer_score(algorithme, method="moyenne custom", parameters=f"x={x}", cb=cb) 

def calculer_score_moyenne():
    x = get_variable('x')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    caracteristique = get_variable('caracteristique')

    if not check_is_defined(x) or not check_is_defined(r_petite_caracteristique) or not check_is_defined(r_grande_caracteristique):
        return

    def algorithme(d):
        k = caracteristique(d)
        if k < x:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
    
    def cb(score):
        validation_score_moyenne()

    calculer_score(algorithme, method="moyenne ref", parameters=f"x={x}", cb=cb) 

def calculer_score_hist_seuil():
    x = get_variable('x')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    caracteristique = get_variable('caracteristique')

    if not check_is_defined(x) or not check_is_defined(r_petite_caracteristique) or not check_is_defined(r_grande_caracteristique):
        return
    
    if x > 35 or x < 33:
        print_error("Trouve un seuil x qui donne un score inférieur à 31% pour continuer.")
        return
    
    def algorithme(d):
        k = caracteristique(d)
        if k < x:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
    
    def cb(score):
        validation_question_hist_seuil()

    calculer_score(algorithme, method="moyenne ref hist", parameters=f"x={x}", cb=cb) 

### ----- CELLULES VALIDATION ----

class ValidatePixelNoir(MathadataValidate):
    def __init__(self):
        super().__init__(success="Bravo, le pixel (17,15) est devenu noir !")

    def validate(self, errors, answers):
        answers['pixel'] = int(d[17][15])
        if d[17][15] == 0:
            return True
        elif d[17][15] == 254:
            errors.append("Tu n'as pas changé la valeur du pixel, il vaut toujours 254")
        else:
            errors.append("Tu as bien changé la valeur mais ce n'est pas la bonne. Relis l'énoncé pour voir la valeur à donner pour un pixel noir.")
        return False

class ValidateScorePixel(MathadataValidateVariables):
    def __init__(self):
        super().__init__({'coordonnees_pixel': None}, success="")
    
    def validate(self, errors, answers):
        if not super().validate(errors, answers):
            return False
        if not check_pixel_coordinates(answers['coordonnees_pixel'], errors):
            return False
        algorithme = get_algorithme_func()
        if not algorithme:
            return False

        def cb(score):
            if score <= 0.4:
                print("Bravo, tu as trouvé un pixel qui te donne un score de 40% ou moins ! Tu peux passer à la suite")
                MathadataValidate()()
            else:
                print_error("Change les coordonnées de ton pixel pour obtenir un score de 40% ou moins.")
                

        calculer_score(algorithme, method="one pixel", cb=cb, a=answers['coordonnees_pixel'], b=answers['coordonnees_pixel']) 
        return False
        

class Validate7(MathadataValidate):
    def validate(self, errors, answers):
        if not has_variable('r_prediction'):
            answers['r_prediction'] = None
            errors.append("La variable r_prediction n'a pas été définie.")
            return False

        r_prediction = get_variable('r_prediction')
        answers['r_prediction'] = r_prediction

        tips = "Vérifie que tes réponses sont bien écrites entre crochets séparées par des virgules comme dans l'exemple : [7,2,2,2,7,2,7,2,2,2]"

        if not isinstance(r_prediction, list):
            errors.append(tips)
            return False
        
        if len(r_prediction) != 10:
            errors.append(f"r_prediction doit contenir 10 valeurs mais ta liste n'en contient que {len(r_prediction)}. {tips}")
            return False

        expected = [estim(d) for d in d_train[0:10]]
        for i in range(10):
            if not isinstance(r_prediction[i], int) or (r_prediction[i] != 2 and r_prediction[i] != 7):
                errors.append(f"Les valeurs de r_prediction doivent être 2 ou 7.")
                return False
            
            if expected[i] != r_prediction[i]:
                errors.append(f"La valeur de r_prediction pour l'image {i + 1} n'est pas la bonne.")
                return False
        return True

class Validate8(MathadataValidate):

    def __init__(self, *args, **kwargs):
        super().__init__(success="")

    def validate(self, errors, answers):
        if not has_variable('e_train_10'):
            errors.append("La variable e_train_10 n'a pas été définie.")
            return False
        
        e_train_10 = get_variable('e_train_10')
        nb_errors = np.count_nonzero(np.array([estim(d) for d in d_train[0:10]]) != r_train[0:10]) 
        if nb_errors * 10 == e_train_10:
            print(f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 premières images, soit {e_train_10}% d'erreur")
            return True
        else:
            if e_train_10 == nb_errors:
                errors.append("Ce n'est pas la bonne valeur. Pour passer du nombre d'erreurs sur 10 images au pourcentage, tu dois multiplier par 10 !")
            elif e_train_10 < 0 or e_train_10 > 100:
                errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
            else:
                errors.append("Ce n'est pas la bonne valeur. Compare ta liste de prédictions avec les vraies valeurs pour trouver le pourcentage d'erreur.")
            return False

if sequence:
    
    @validationclass
    class ValidatePixelNoir(ValidatePixelNoir):
        pass

    @validationclass
    class ValidateScorePixel(ValidateScorePixel):
        pass

    @validationclass
    class Validate7(Validate7):
        pass

    @validationclass
    class Validate8(Validate8):
        pass



validation_score_pixel = ValidateScorePixel()

validation_question_3 = MathadataValidateVariables({
    'r': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "Tu dois répondre par 2 ou 7."
            }
        ]
    }
})
validation_question_5 = MathadataValidateVariables({'r_petite_caracteristique': 7, 'r_grande_caracteristique': 2}, success="Bravo, la moyenne est en effet plus élevée pour les images de 2 que de 7")
validation_question_6 = MathadataValidateVariables({
    'x': {
        'value': {
            'min': 23,
            'max': 57
        }
    }
}, success="Ton seuil est correct ! Il n'est pas forcément optimal, on verra dans la suite comment l'optimiser.")

validation_question_7 = Validate7()
validation_question_8 = Validate8()
validation_question_9 = MathadataValidateVariables({
  'x': {
      'value': {
            'min': 32,
            'max': 36
      }
  }  
})

validate_moyenne_partie_image = MathadataValidateVariables({
    'e_train': {
        'value': {
            'min': 0,
            'max': 12
        },
        'errors': [
            {
                'value': {
                    '<': 0
                },
                'else': "Continue à chercher une zone qui différencie bien les 2 et les 7. Pour cela remonte à la cellule qui permet de changer les coordonnées des points A et B. Choisi une rectangle rouge qui diférencie bien les 2 et les 7."
            }
        ]
    }
}, success="Bravo ! Tu as réussi à réduire l'erreur à moins de 12%. C'est déjà un très bon score ! Tu peux continuer à essayer d'améliorer ta zone ou passer à la suite.")

## Validation des questions Stat

# Les success
def on_success_hist_1(answers):
    afficher_histogramme(caracteristique)

validation_question_hist_1 = MathadataValidateVariables({
    'r_histogramme_orange': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "r_histogramme_orange n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    },
    'r_histogramme_bleu': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "r_histogramme_bleu n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    }
}, success="C'est la bonne réponse ! Les images de 7 ont souvent moins de pixels blancs que les images de 2. C'est pourquoi leur caractéristique est souvent plus petite.", on_success=on_success_hist_1)

validation_question_hist_2 = MathadataValidateVariables({
    'nombre_2': {
        'value': {
            'min': 47,
            'max': 50
        },
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 350
                },
                'else': "nombre_2 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 2 avec une caractéristique entre 20 et 22 ?"
            }
        ]
    },
    'nombre_7': {
        'value': {
            'min': 220,
            'max': 240
        },
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 350
                },
                'else': "nombre_7 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 7 avec une caractéristique entre 20 et 22 ?"
            }
        ]
    },
}, success="C'est la bonne réponse !")

validation_question_hist_3 = MathadataValidateVariables({
    'nombre_2_inf_20': {
        'value': {
            'min': 50,
            'max': 70
        },
        'errors': [
            {
                'value': {
                    'min': 10,
                    'max': 100
                },
                'else': "nombre_2_inf_20 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 2 avec une caractéristique inférieure à 20 ?"
            }
        ]
    },
    'nombre_7_inf_20': {
        'value': {
            'min': 290,
            'max': 330
        },
        'errors': [
            {
                'value': {
                    'min': 100,
                    'max': 400
                },
                'else': "nombre_7_inf n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 7 avec une caractéristique inférieure à 20 ?"
            }
        ]
    },
}, success="C'est la bonne réponse !")

validation_question_hist_seuil = MathadataValidateVariables({
    'x': {
        'value': {
            'min': 33,
            'max': 35
        }
    }
}, success="Bravo, ton seuil est maintenant optimal !")

### Pour les checks d'execution des cellules sans réponse attendue:
validation_execution_1 = MathadataValidate(success="")
validation_execution_2 = MathadataValidate(success="")
validation_execution_3 = MathadataValidate(success="")
validation_execution_4 = MathadataValidate(success="")
validation_execution_5 = MathadataValidate(success="")
validation_execution_6 = MathadataValidate(success="")
validation_execution_7 = MathadataValidate(success="")
validation_execution_classif = MathadataValidate(success="")
validation_execution_caracteristique_custom = MathadataValidate(success="")
validation_score_moyenne = MathadataValidate(success="")
validation_score_zone_custom = MathadataValidate(success="")