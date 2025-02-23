import os
import sys
import matplotlib.pyplot as plt
import requests
import secrets
import __main__
import json

### --- AJOUT DE TOUS LES SUBDIRECTIRIES AU PATH ---
base_directory = os.path.abspath('.')

# Using os.listdir() to get a list of all subdirectories
subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Adding all subdirectories to the Python path
sys.path.extend(subdirectories)


### --- IMPORT DE BASTHON ---
# Ne marche que si on est sur basthon ou capytale, sinon ignorer : 
try:
    import basthon  # Ne marche que si on est sur Capytale ou Basthon
    basthon = True

except ModuleNotFoundError: 
    basthon = False
    pass

### --- Import du validation_kernel ---
# Ne marche que si fourni et si va avec le notebook en version séquencé. Sinon, ignorer :
sequence = False

try:
    from capytale.autoeval import Validate, validationclass
    from capytale.random import user_seed

    sequence = True
except ModuleNotFoundError: 
    sequence = False

## Pour valider l'exécution d'une cellule de code, dans le cas du notebook sequencé :
if sequence:
    validation_execution = Validate()
else:
    def validation_execution():
        return True

# Fonctions avec comportement different sur Capytale ou en local

if sequence:
    from js import fetch, FormData, Blob, eval, Headers
    import asyncio
    
    debug = False
    analytics_endpoint = "https://dev.mathadata.fr/api/notebooks"
    challengedata_endpoint = "https://challengedata.ens.fr/api"
    
    Validate()() # Validate import cell

    def call_async(func, cb, *args):
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(func(*args))

            def internal_cb(future):
                try:
                    data = future.result()
                    if cb is not None:
                        cb(data)
                except Exception as e:
                    if debug:
                        print_error("Error during post request")
                        print_error(e)
                        
            task.add_done_callback(internal_cb)
        except Exception as e:
            if debug:
                print_error("Error during post request")
                print_error(e)

    async def fetch_async(url, method='GET', body=None, files=None, fields=None, headers=None):
        if body:
            body = json.dumps(body)
        elif files or fields:
            body = FormData.new()
            if files:
                for key in files:
                    body.append(key, Blob.new([files[key].getvalue()], {'type' : 'text/csv'}))
            if fields:
                for key in fields:
                    body.append(key, str(fields[key]))
                    
        if headers:
            js_headers = Headers.new()
            for key in headers:
                js_headers.append(key, headers[key])
            
        response = await fetch(url, method=method, body=body, headers=js_headers)
        if response.status >= 200 and response.status < 300:
            data = await response.text()
            data_json = json.loads(data)
            return data_json
        else:
            raise Exception(f"Fetch failed with status: {response.status}")

    # Récupère l'id capytale, le statut prof/élève et le nom de classe depuis l'API capytale
    async def get_profile():
        return await fetch_async('/web/c-auth/api/me?_format=json')
    
else:
    from IPython.display import display, Javascript

    debug = True
    analytics_endpoint = "http://localhost:3000/api/notebooks"
    challengedata_endpoint = "http://localhost:8000/api"
    #analytics_endpoint = "https://dev.mathadata.fr/api/notebooks"


### Utilitaires requêtes HTTP ###

# Send HTTP request. To send body as form data use parameter files (dict of key, StringIO value) and fields (dict of key, value)
def http_request(url, method='GET', body=None, files=None, fields=None, headers=None, cb=None):
    if body is not None and (files is not None or fields is not None):
        raise ValueError("Cannot have both body and files in the same request")
    try:
        if sequence:
            call_async(fetch_async, cb, url, method, body, files, fields, headers)
        else:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=body, files=files, data=fields, headers=headers)
            elif debug:
                raise ValueError(f"Invalid method: {method}")
            else:
                return None
        
            if cb is not None:
                cb(response.json())
                
    except Exception as e:
        if debug:
            print_error("Error during http request")
            print_error(e)

challengedata_token = 'yjQTYDk8d51Uq8WcDCPUBK1GPEuEDi6W/3e736TV7qGAmmqn7CCyefkdL+vvjOFY'

def http_request_cd(endpoint, method='GET', body=None, files=None, fields=None, cb=None):
    headers = {
        'Authorization': f'Bearer {challengedata_token}'
    }
    http_request(challengedata_endpoint + endpoint, method, body, files, fields, headers, cb)


### Analytics ###

session_id = None
capytale_id = None
capytale_classroom = None

def start_analytics_session(notebook_id):
    global capytale_id, capytale_classroom
    if sequence:
        seed = user_seed()
        if seed == 609507 or seed == 609510:
            capytale_id = seed
            capytale_classroom = "dev_capytale"
            debug = True
            get_highscore()
            return
        
        def profile_callback(profile):
            try:
                capytale_id = profile['uid']
                capytale_classroom = profile['classe']
                get_highscore()
                if profile['profil'] == 'teacher':
                    return

                # create analytics session for students except mathadata accounts
                create_session(notebook_id)
            except Exception as e:
                if debug:
                    print_error("Error during post request")
                    print_error(e)
        
        call_async(get_profile, profile_callback)
        
    else:
        capytale_id = -1
        capytale_classroom = "dev"
        create_session(notebook_id)
        get_highscore()


def create_session(notebook_id):
    def cb(data):
        global session_id
        session_id = data['id']

    http_request(analytics_endpoint + '/session', 'POST', {
        'notebook_id': notebook_id,
        'user_id': capytale_id,
        'classname': capytale_classroom
    }, cb=cb)


### Gestion du score ###

highscore = None
session_score = None

# Display score as percentage with one decimal
def score_str(score):
    percent = score * 100
    return f"{percent:.1f}%"

def get_highscore(challenge_id=116):
    def cb(data):
        if data is not None and isinstance(data, dict) and 'highscore' in data:
            global highscore
            highscore = data['highscore']
            update_score()
        elif debug:
            print_error("Failed to get highscore. Received data" + str(data))
        
    http_request_cd(f'/participants/challenges/{challenge_id}/highscore?capytale_id={capytale_id}', 'GET', cb=cb)

def set_score(score):
    print('Nouveau score : ' + score_str(score))
    global session_score, highscore
    if session_score is None or score < session_score:
        session_score = score
        if highscore is None or session_score < highscore:
            highscore = session_score
        update_score()
    
    
def submit(csv_content, challenge_id=116, method=None, parameters=None, cb=None):
    if (capytale_id is None):
        return
    
    def internal_cb(data):
        if data is not None and isinstance(data, dict) and 'score' in data:
            set_score(data['score'])
            
            if cb is not None:
                cb(data['score'])
        else:
            print('Il y a eu un problème lors du calcul du score. Ce n\'est probablement pas de ta faute, réessaye dans quelques instants.')
            if debug:
                print_error("Received data" + str(data))

    http_request_cd(f'/participants/challenges/{challenge_id}/submit', 'POST', files={
        'file': csv_content,
    }, fields={
        'method': method,
        'parameters': parameters,
        'capytale_id': capytale_id,
        'capytale_classroom': capytale_classroom,
    }, cb=internal_cb)
    
    
### Customisation page web ###

def run_js(js_code):
    if sequence:
        eval(js_code)
    else:
        display(Javascript(js_code))

# Fonctions utilitaires
run_js("""
    const mathadata = {
        import_js_script(url, callback) {
            if (window.require) {
                require([url], callback);
            } else {
                var script = document.createElement('script');
                script.src = url;
                script.onload = callback;
                document.head.appendChild(script);
            }
        }
    }

    window.mathadata = mathadata;
""")


def create_sidebox():
    js_code = f"""
    
    let style = document.getElementById('sidebox-style');
    if (style !== null) {{
        style.remove();
    }}

    style = document.createElement('style');
    style.id = 'sidebox-style';
    style.innerHTML = `
        #sidebox {{
            position: fixed;
            top: 20vh;
            left: 0;
            max-height: 60vh;
            width: 20vw;
            background-color: white;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            transition: left 0.5s;
        }}

        .sidebox-main {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px;
            flex: 1;
        }}

        .sidebox-collapse-button {{
            position: absolute;
            left: 100%;
            top: 30%;
            bottom: 30%;
            margin: auto;
            padding: 0.5rem;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            background-color: black;
            border-left: none;
            border-radius: 0 5px 5px 0;
        }}
        
        .sidebox-header {{
            display: flex;
            gap: 3px;
            align-items: center;
        }}
        
        .sidebox-section {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
            align-items: center;
        
        }}
        
        .score {{
            font-size: 2rem;
            font-weight: bold;
        }}
    `;
    document.head.appendChild(style);
    
    
    let sidebox = document.getElementById('sidebox');
    if (sidebox !== null) {{
        sidebox.remove();
    }}
    
    sidebox = document.createElement('div');
    sidebox.id = 'sidebox';
    sidebox.style.left = '-20vw';
    
    sidebox.innerHTML = `
        <div class="sidebox-main">
            <div class="sidebox-header">
                <h3>Score</h3>
            </div>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around; width: 100%; margin-top: 2rem">
                <div class="sidebox-section">
                    <h4>Meilleur</h4>
                    <svg xmlns="http://www.w3.org/2000/svg" width=40 height=40 viewBox="0 0 512 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#B197FC" d="M4.1 38.2C1.4 34.2 0 29.4 0 24.6C0 11 11 0 24.6 0H133.9c11.2 0 21.7 5.9 27.4 15.5l68.5 114.1c-48.2 6.1-91.3 28.6-123.4 61.9L4.1 38.2zm503.7 0L405.6 191.5c-32.1-33.3-75.2-55.8-123.4-61.9L350.7 15.5C356.5 5.9 366.9 0 378.1 0H487.4C501 0 512 11 512 24.6c0 4.8-1.4 9.6-4.1 13.6zM80 336a176 176 0 1 1 352 0A176 176 0 1 1 80 336zm184.4-94.9c-3.4-7-13.3-7-16.8 0l-22.4 45.4c-1.4 2.8-4 4.7-7 5.1L168 298.9c-7.7 1.1-10.7 10.5-5.2 16l36.3 35.4c2.2 2.2 3.2 5.2 2.7 8.3l-8.6 49.9c-1.3 7.6 6.7 13.5 13.6 9.9l44.8-23.6c2.7-1.4 6-1.4 8.7 0l44.8 23.6c6.9 3.6 14.9-2.2 13.6-9.9l-8.6-49.9c-.5-3 .5-6.1 2.7-8.3l36.3-35.4c5.6-5.4 2.5-14.8-5.2-16l-50.1-7.3c-3-.4-5.7-2.4-7-5.1l-22.4-45.4z"/></svg>
                    <span id="highscore" class="score">{score_str(highscore) if highscore is not None else "..."}</span>
                </div>
                <div class="sidebox-section">
                    <h4>Session</h4>
                    <svg xmlns="http://www.w3.org/2000/svg" width=40 height=40 viewBox="0 0 448 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#63E6BE" d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/></svg>
                    <span id="session-score" class="score">...</span>
                </div>
            </div>
        </div>
    `;

    let collapseButton = document.createElement('div');
    collapseButton.className = 'sidebox-collapse-button';
    collapseButton.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" width=24 height=24><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#FFD43B" d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"/></svg>
    `;
    collapseButton.addEventListener('click', () => {{
        let sidebox = document.getElementById('sidebox');
        if (sidebox.style.left === '0px') {{
            sidebox.style.left = '-20vw';
        }} else {{
            sidebox.style.left = '0px';
        }}
    }});

    sidebox.appendChild(collapseButton);
    document.body.appendChild(sidebox);
    """    
    
    run_js(js_code)

# Exécuté à l'import du notebook
create_sidebox()

def update_score():
    if highscore is None and session_score is None:
        return 
    js_code = ""
    if highscore is not None:
        js_code += f"document.getElementById('highscore').innerText = '{score_str(highscore)}';"
    if session_score is not None:
        js_code += f"document.getElementById('session-score').innerText = '{score_str(session_score)}';"

    # Pop out the sidebox if the score is updated
    js_code += "document.getElementById('sidebox').style.left = '0';"

    run_js(js_code)

### VALIDATION CLASSES ###


class _MathadataValidate():
    counter = 0
    
    def __init__(self, success=None, function_validation=None, on_success=None, *args, **kwargs):
        self.trials = 0
        self.success = success
        self.function_validation = function_validation
        self.child_on_success = on_success

    def __call__(self):

        # Set question number at first trial
        if self.trials == 0:
            _MathadataValidate.counter += 1
            self.question_number = _MathadataValidate.counter

        self.trials += 1
        errors = []
        answers = {}

        res = None
        try:
            res = self.validate(errors, answers)
        except Exception as e:
            errors.append("Il y a eu une erreur dans la validation de la question. Vérifie que ta réponse est écrite correctement")
            if debug:
                errors.append(str(e))

        if res is None:
            res = len(errors) == 0

        self.send_analytics_event(res, answers)

        if len(errors) > 0:
            for error in errors:
                print_error(error)


        if res:
            self.on_success(answers) 

        return res

    def send_analytics_event(self, res, answers):
        if session_id is not None:
            try:
                answers_json = json.dumps(answers)
            except TypeError as e:
                answers = {}

            http_request(analytics_endpoint + '/event', 'POST', {
                'session_id': session_id,
                'question_number': self.question_number,
                'is_correct': res,
                'answer': answers,
            })

    def validate(self, errors, answers):
        if self.function_validation is not None:
            return self.function_validation(errors, answers)
        return True

    def on_success(self, answers):
        if self.success is not None:
            if self.success:
                print(self.success)
        else:
            print("Bravo, c'est la bonne réponse !")
        if self.child_on_success is not None:
            self.child_on_success(answers)

    def trial_count(self):
        return self.trials

if sequence:

    @validationclass
    class MathadataValidate(_MathadataValidate, Validate):
        
        def __init__(self, *args, **kwargs):
            _MathadataValidate.__init__(self, *args, **kwargs)
            Validate.__init__(self)
else:
    MathadataValidate = _MathadataValidate

def check_is_defined(var):
    # let's check of the local variable var is defined and not Ellipsis:
    if var is Ellipsis:
        print_error(f"As-tu bien remplacé les ... ? Une variable n'est pas définie.")
        return False
    else:
        return True
 
class MathadataValidateVariables(MathadataValidate):
    def __init__(self, name_and_values, *args, **kwargs):
        self.name_and_values = name_and_values
        super().__init__(*args, **kwargs)

    def check_undefined_variables(self, errors):
        undefined_variables = [name for name in self.name_and_values if not has_variable(name) or get_variable(name) is Ellipsis]
        undefined_variables_str = ", ".join(undefined_variables)

        if len(undefined_variables) == 1:
            errors.append(f"As-tu bien remplacé les ... ? La variable {undefined_variables_str} n'a pas été définie.")
        elif len(undefined_variables) > 1:
            errors.append(f"As-tu bien remplacé les ... ? Les variables {undefined_variables_str} n'ont pas été définies.")

        return undefined_variables
    
    def check_variables(self, errors):
        for name in self.name_and_values:
            val = get_variable(name)
            expected = self.name_and_values[name]
            if expected is None: # Variable is optional
                res = True
            elif isinstance(expected, dict):
                res = self.compare(val, expected['value'])
            else:
                res = self.compare(val, expected)

            if not res:
                self.check_errors(errors, name, val, expected)
        
    def check_type(self, errors, name, val, expected, tips=None):
        if type(val) != type(expected):
            errors.append(f"La variable {name} n'est pas du bon type. Le type attendu est {type(expected)} mais le type donné est {type(val)}.")
            return False
        else:
            return True
    
    def compare(self, val, expected):
        try:
            if isinstance(expected, dict):
                if 'is' in expected:
                    return val == expected
                elif 'min' in expected and 'max' in expected:
                    return val >= expected['min'] and val <= expected['max']
                elif 'in' in expected:
                    return val in expected['in']
                else:
                    raise ValueError(f"Malformed validation class : comparing {val} to {expected}")
            else:
                return val == expected
        except Exception as e:
            return False

    def check_errors(self, errors, name, val, expected):
        if isinstance(expected, dict) and 'errors' in expected:
            for error in expected['errors']:
                match_error = self.compare(val, error['value'])
                if match_error and 'if' in error:
                    errors.append(error['if'])
                    return
                if not match_error and 'else' in error:
                    errors.append(error['else'])
                    return

        # Default error message
        errors.append(f"{name} n'a pas la bonne valeur.")

    def validate(self, errors, answers):
        for name in self.name_and_values:
            if not has_variable(name):
                answers[name] = None
            else:
                answers[name] = get_variable(name)

        undefined_variables = self.check_undefined_variables(errors)
        if len(undefined_variables) == 0:
            if self.function_validation is not None:
                return self.function_validation(errors, answers)
            else:
                self.check_variables(errors)

        return len(errors) == 0

if sequence:

    @validationclass
    class MathadataValidateVariables(MathadataValidateVariables):
        pass


### --- Config matplotlib ---

plt.rcParams['figure.dpi'] = 150


### --- Common util functions ---

def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def has_variable(name):
    return hasattr(__main__, name) and get_variable(name) is not None and get_variable(name) is not Ellipsis

def get_variable(name):
    return getattr(__main__, name)



### ONGOING ###

def create_exercice(div_id, config):
    
    answers_html = ''
    for answer in config['answers']:
        if answer['type'] == 'number':
            answers_html += f'<label for="{div_id}-answer-{answer["name"]}">{answer["name"]}&nbsp;:&nbsp;</label>'
            answers_html += f'<input type="number" id="{div_id}-answer-{answer["name"]}" style="width: 100px; margin-top: 1rem; padding: 0.5rem; font-size: 1rem">'

    answers_html += '<button id="submit-button" style="margin-top: 1rem; padding: 0.5rem; font-size: 1rem">Valider</button>'
    
    js_code = f"""
    // clear div content
    const div = document.getElementById('{div_id}');
    if (div !== null) {{
        div.innerHTML = '';
    }}
    
    div.innerHTML = `
        <div>{config['question']}</div>
        <div><canvas id="{div_id}-canvas"></canvas></div>
        <div>{answers_html}</div>
    `;
    
    const canvas = document.getElementById('{div_id}-canvas');
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {json.dumps(config['chart'])});
    """
    run_js(js_code)

# create_exercice('exercice', {
#     'question': 'Donnez les paramètres m et p de la droite d\'équation y = mx + p qui passe par les points A et B',
#     'chart': {
#         'type': 'scatter',
#         'data': {
#             'labels': ['A', 'B'],
#             'datasets': [{
#                 'label': 'Points',
#                 'data': [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}],
#                 'fill': False,
#                 'borderColor': 'rgb(75, 192, 192)',
#                 'lineTension': 0.1
#             }]
#         }
#     },
#     'answers': [
#         {'name': 'm', 'type': 'number'},
#         {'name': 'p', 'type': 'number'},
#     ],

# })