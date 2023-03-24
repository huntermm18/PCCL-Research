import argparse
from util.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', action='store_true')
args = parser.parse_args()


# ========================= Parameters ===================================================
PROMPT = """
        When taking a survey, a {category} replied to the following question. The question said:
         "What is your favorite kind of cookie?"
        The response options are:
        A. Sugar Cookie
        B. Chocolate Cookie
        C. Oatmeal Cookie

        They responded with: """

CHOICES = ['A. Sugar Cookie', 'B. Chocolate Cookie', 'C. Oatmeal Cookie']
CATEGORIES = ['man', 'woman']
ITERATIONS = 70
X_LABEL = 'Gender'  # For graph
TEMPERATURE = 1.0
MAX_TOKENS = 4

# =====================================================================================

if __name__ == '__main__':
    if not args.i:
        run_survey(PROMPT,
                   choices=CHOICES,
                   categories=CATEGORIES,
                   iterations=ITERATIONS,
                   x_label=X_LABEL,
                   max_tokens=MAX_TOKENS,
                   temperature=TEMPERATURE)
    else:
        from colorama import Fore
        print('Interactive Mode...\n')
        print(Fore.CYAN + 'Leave any prompt blank to take the value currently in the file\n')
        # print(Fore.WHITE)
        prompt = input(Fore.CYAN + 'Enter a prompt: ' + Fore.RESET)
        if not prompt:
            prompt = PROMPT
        choices = input(Fore.CYAN + 'Enter selection choices seperated by comma (A. Sugar Cookie, B. Chocolate Cookie, C. Oatmeal Cookie): ' + Fore.RESET)
        if not choices:
            choices = CHOICES
        else:
            choices = choices.split(',')
            choices = [c.strip() for c in choices]
        categories = input(Fore.CYAN + 'Enter categories seperated by comma (e.g. man, woman): ' + Fore.RESET)
        if not categories:
            categories = CATEGORIES
        else:
            categories = categories.split(',')
            categories = [c.strip() for c in categories]
        iterations = input(Fore.CYAN + 'Enter iterations: ' + Fore.RESET)
        if not iterations:
            iterations = ITERATIONS
        x_label = input(Fore.CYAN + 'Enter an X-axis label for the graph: ' + Fore.RESET)
        if not x_label:
            x_label = X_LABEL
        max_tokens = input(Fore.CYAN + 'Enter max tokens: ' + Fore.RESET)
        if not max_tokens:
            max_tokens = MAX_TOKENS
        temperature = input(Fore.CYAN + 'Enter model temperature: ' + Fore.RESET)
        if not temperature:
            temperature = TEMPERATURE

        print(f'Prompt: {prompt[:50].strip()}...\nChoices: {choices}\nCategories: {categories}\nIterations: {iterations}\nX Label: {x_label}\nMax Tokens: {max_tokens}\nTemperature: {temperature}')

        # Run survey
        print('Running survey...\n')
        run_survey(prompt,
                   choices=choices,
                   categories=categories,
                   iterations=int(iterations),
                   x_label=x_label,
                   max_tokens=int(max_tokens),
                   temperature=float(temperature))
        print('Done')

