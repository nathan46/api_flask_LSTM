
# Projet de détection de violence dans des tweets

SafeTweet est une application mobile pour Android qui permet à un utilisateur de Twitter de naviguer en toute sécurité sur ce-dit réseau social. L’application est un Twitter-like, c’est-à-dire, une application semblable à Twitter. Elle permet à son utilisateur de consulter sa timeline - Fil d’actualités - et d’informer (voire filtrer) le contenu qui peut s’avérer dangereux. Le domaine d’application des fake news étant très vaste, l’application a alors pour but de rassurer ou/et de prévenir les personnes victimes de harcèlement.

Pour faire cela nous avons fait :
 - Une application android : https://github.com/nouveliere-benjamin/Fake-News-2019_2020
 - Un notebook qui regroupe la création et l'entrainement du modèle LSTM ainsi que la Dataset utilise : https://github.com/nathan46/detection_tweet_violent_LSTM
 - Une api python qui permet de faire le lien entre les deux : https://github.com/nathan46/api_flask_LSTM

# Api Python
Cette Api est une api Flask qui permet de relier notre application android à notre IA.

## Pour commencer

### Cloner le dépôt
```bash
$ git clone https://github.com/nathan46/api_flask_LSTM.git
$ cd api_flask_LSTM
```
### Installer les dépendances
```bash
$ pip3 install -r requirements.txt
```

### Lancer l'application
```bash
$ sudo python3 apiFlask.py
```
Normalement vous devriez avoir ceci :
```bash
[nltk_data] Downloading package stopwords to /home/nathan/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
* Serving Flask app "apiFlask" (lazy loading)
* * Environment: production
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
* Debug mode: off
* Running on http://0.0.0.0:80/ (Press CTRL+C to quit)
```

### Pour communiquer avec l'api

Vous pouvez seulement faire des requêtes POST avec les données en JSON structurées de cette manière :
```json
{"tweet": "Un exemple de tweet a tester"}
```
Vous ne pouvez analyser qu'un seul tweet par requête.
Vous recevrez une réponse en JSON structurée comme cela :
```json
{"score":0.7452720999717712}
```
Exemple de requête :
```bash
$ curl localhost -d '{"tweet": "Merci beaucoup pour votre soutient vous etes tres gentil"}' -H 'Content-Type: application/json'
```
Réponse :
```json
{"score":0.07911539822816849}
```

Plus le score est proche de 0 plus il est considéré comme gentil/neutre alors qu’au contraire un tweet classifié proche de 1 est considéré comme méchant/menaçant/vulgaire

## Différentes configurations

### Changer de modèle
Vous pouvez charger des modèles différents, pour cela il faut générer un nouveau modèle grâce à ce dépôt :
https://github.com/nathan46/detection_tweet_violent_LSTM

Une fois les fichiers générés, placer le fichier **model_unNomAuHasard.pth** dans le dossier **model**
Puis le fichier **vocab_to_int-unNomAuHasard.json** dans le dossier **vocab_to_int**

### Charger le modèle dans le code
Il suffit de changer la variable **nom_fichier** avec le nom que vous avez donné, par exemple :
```python
nom_fichier = 'unNomAuHasard'
```

Lors de la création du modèle (notebook), si vous avez changé des variables qui font varier l'architecture du modèle, par exemple **n_layers**, **hidden_dim** etc.
Il ne faudra pas oublier de les reporter aussi dans le code de l'API.




