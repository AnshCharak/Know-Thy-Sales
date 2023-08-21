1) Colab Files folder contains the ipynb files which include data preprocessing, model training, testing, etc.
   The model used in the web application (final product) use the same technique

2) Final Product folder contains the django web applcation. Steps for running it are:
	i)   Install Python 3.7 or higher
	ii)  Install the django framework using pip install django 
	iii) Install packages numpy, sklearn, pandas, scikit-learn
	iv)  Using terminal, go to the directory containing the manage.py file in the final product folder
	v)   Type command->  python manage.py runserver
	vi)  Using a browser open URL -> http://127.0.0.1:8000/analytics/	