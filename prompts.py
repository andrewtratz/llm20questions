
import string
import unicodedata

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def normalize(s):
    t = str.maketrans("", "", string.punctuation)
    return strip_accents(s.lower().replace("the", "").replace(" ", "").translate(t))

class Prompter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_from_messages(self, messages, temp=1.4, max_tokens=256):    
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    # Base wrapper for all system prompts
    def sys_wrapper(self, sys_prompt):
        messages = []
        sys_message = {"role": "system"}
        sys_message["content"] = sys_prompt
        messages.append(sys_message)
        return messages

    # Combine prompts and run messages
    def run_prompts(self, messages, prompt, temp=1.4, DEBUG=False):        
        usr_message = {"role": "user", "content": prompt}
        if DEBUG:
            print(prompt + '\n') 
        messages.append(usr_message)
        output = self.generate_from_messages(messages, temp)
        if DEBUG:
            print(output + '\n')
        return output
        
    # System prompt for standard list-maker
    def sys_listmaker(self):
        return self.sys_wrapper("""You are a helpful AI assistant who is skilled in creating diverse lists of objects
        and asking questions to subdivide these objects into separate categories.\n""")

    def sys_rephrase(self):
        return self.sys_wrapper("""You are a helpful AI assistant who is skilled at rephrasing questions. 
        You respect the integrity and meaning of the original sentence and do in-place substitutions only with no hallucinations."""   ) 

    def sys_answerer(self):
        return self.sys_wrapper("""You are a helpful AI assistant who is skilled in at answering yes-no-questions. 
        You are highly accurate and demonstrate a strong understanding of question nuances.""")

    def sys_grammar_editor(self):
        return self.sys_wrapper("""You are a helpful AI assistant who is skilled at editing and understanding grammar, sentence structure, and syntax. """)

    # Creates diverse lists of 30 things.
    def list_of_thirty(self, pos_plural, prior_pos_plural, location, size, questions=None, answers=None, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        
        ask_prompt = "Create a list of 30 " + prior_pos_plural + " that are " 
        if len(size) > 0:
            ask_prompt += size + " "
        ask_prompt += pos_plural 
        if len(location) > 0:
            ask_prompt += "typically located in or at " + location
        ask_prompt += ". "
        #ask_prompt += "The things must be unique, diverse, " + pos_plural + ", and as different as possible from each other. "
        ask_prompt += "The " + pos_plural + " must be common examples of " + pos_plural + " and representative of a range of different possible options. "
        # ask_prompt += "The things must represent examples of as many different categories of " + pos_plural + " as possible. "
        ask_prompt += "The answer should be returned as a comma-separated list with no additional verbose output. "
        ask_prompt += "None of the " + pos_plural + " may be repeated. "
        # ask_prompt += "Each item in the list should be as different as possible from the prior item. "
        ask_prompt += "No words in the list may be repeated. "
        ask_prompt += "Each of the " + pos_plural + " in the list ABSOLUTELY MUST MEET EACH the following criteria:\n"
        for i, q, a in zip(range(0, len(questions)),questions, answers):
            ask_prompt += " " + str(i+1) + ". " + q + " " + a + ".\n"
        ask_prompt += " Respond with the comma-separated list only. Order the responses according to the most likely or common choices according to the criteria above."

        return self.run_prompts(messages, ask_prompt, temp, DEBUG)

    # Creates diverse lists of 30 things.
    def list_of_thirty_withsummary(self, pos_plural, prior_pos_plural, location, material, size, summary, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()

        ask_prompt = "Create a list of 30 " + prior_pos_plural + " that are " 
        if len(size) > 0:
            ask_prompt += size + " "
        ask_prompt += pos_plural 
        if material is not None:
            ask_prompt += " mostly made of " + material
        if len(location) > 0:
            ask_prompt += " typically located in or at " + location
        ask_prompt += ", "
        ask_prompt += f"matching the following description: {summary}\n"
        #ask_prompt += "The things must be unique, diverse, " + pos_plural + ", and as different as possible from each other. "
        ask_prompt += "The " + pos_plural + " must be common examples of " + pos_plural + " and representative of a range of different possible options. "
        # ask_prompt += "The things must represent examples of as many different categories of " + pos_plural + " as possible. "
        ask_prompt += "The answer should be returned as a comma-separated list with no additional verbose output. "
        ask_prompt += "None of the " + pos_plural + " may be repeated. "
        # ask_prompt += "Each item in the list should be as different as possible from the prior item. "
        ask_prompt += "No words in the list may be repeated. "
        # ask_prompt += "Each of the " + pos_plural + " in the list ABSOLUTELY MUST MEET the following description:\n"
        ask_prompt += "Respond with the comma-separated list only. Order the responses according to the most likely or common choices according to the criteria above."

        return self.run_prompts(messages, ask_prompt, temp, DEBUG)

    # Thirty diverse locations question maker
    def list_of_thirty_geo(self, type, plural, questions=None, answers=None, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        ask_prompt = "Create a list of 30 geographic " + plural + ". "
        ask_prompt += "The " + plural + " must be unique, diverse, and as different as possible from each other. "
        ask_prompt += "The things must represent examples of as many different categories or geographies of " + plural + " as possible. "
        ask_prompt += "The answer should be returned as a comma-separated list with no additional verbose output. "
        ask_prompt += "None of the " + plural + " may be repeated. "
        ask_prompt += "Each " + type + " in the list should be as different as possible from the prior " + type + ". "
        ask_prompt += "No words in the list may be repeated. "
        ask_prompt += "Each of the " + plural + " in the list should meet the following criteria:"
        for i, q, a in zip(range(0, len(questions)),questions, answers):
            ask_prompt += " " + str(i+1) + ". " + q + " " + a + "."
        ask_prompt += " Respond with the comma-separated list only."
        return self.run_prompts(messages, ask_prompt, temp, DEBUG)

    def split_category(self, prior_pos_plural, plural, negative_categories, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = 'Divide the category "' + prior_pos_plural + ' that are ' + plural + '" into two broad, clearly-defined, non-overlapping sub-categories,'
        prompt += 'ensuring that all ' + prior_pos_plural + ' that are ' + plural + ' fall into one category or the other, but not both. '
        prompt += 'Respond with the names of the two sub-categories separated by a comma only. Do NOT repeat the original category. Use common phrasing. '
        prompt += 'Each sub-category must be a noun or noun phrase.'
        if len(negative_categories) > 0:
            prompt += " Do not use any of the following sub-categories: "
            for cat, i in zip(negative_categories, range(0, len(negative_categories))):
                prompt += cat
                if i < len(negative_categories)-1:
                    prompt += ", "
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def split_category_withsummary(self, summary, negative_categories, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = 'Divide the category described as "' + summary + '" into two broad, clearly-defined, non-overlapping sub-categories,'
        prompt += 'ensuring that all members of the category fall into one category or the other, but not both. '
        prompt += 'Respond with the names of the two sub-categories separated by a comma only. Do NOT repeat the original category. Use common phrasing. '
        prompt += 'Each sub-category must be a noun or noun phrase.'
        if len(negative_categories) > 0:
            prompt += " Do not use any of the following sub-categories: "
            for cat, i in zip(negative_categories, range(0, len(negative_categories))):
                prompt += cat
                if i < len(negative_categories)-1:
                    prompt += ", "
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def question_category(self, category, temp=1.4, DEBUG=False):
        messages = self.sys_grammar_editor()
        # if part_of_speech == 'noun':
        #     particle = 'a '
        # else:
        #     particle = ''
        # prompt = 'Rephrase the question "Is it a '+ category + '?" '
        prompt = 'Rephrase the question "Is it commonly and often described as a '+ category + '?" '
        prompt += " to use correct grammar, including modification of plurality, particle, or gender. Do not change any of the key words or concepts. "
        prompt += "The question should be a simple yes or no question. Respond with the question only without any additional introduction or conclusion."
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def question_location(self, location, temp=1.4, DEBUG=False):
        messages = self.sys_grammar_editor()
        # if part_of_speech == 'noun':
        #     particle = 'a '
        # else:
        #     particle = ''
        prompt = 'Rephrase the question "Are they typically found in or at ' + location + '?"'
        prompt += " to use correct grammar, including modification of plurality, particle, or gender. Do not change any of the key words or concepts. "
        prompt += "The question should be a simple yes or no question. Respond with the question only without any additional introduction or conclusion."
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def noun_phrases(self, word, exclude=[], temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = f"Create 20 noun phrases describing tangible things that begin with the word '{word}'. Each noun phrase should be exactly two words long, and must begin with the word '{word}' followed by a space without any modification. Respond with the noun phrases in a comma-separated list with no introduction or other additional text."
        if len(exclude) > 0:
            prompt += " Do not include any of the following: '" + "', '".join(exclude) + "'."
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def tangible_object(self, phrase, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = f"Is '{phrase}' typically a tangible thing that can be seen or felt? Respond with a single word yes or no, without any introduction or additional text. Do not add punctuation to the answer."
        return normalize(self.run_prompts(messages, prompt, temp, DEBUG))

    def question_size(self, size, temp=1.4, DEBUG=False):
        messages = self.sys_grammar_editor()
        # if part_of_speech == 'noun':
        #     particle = 'a '
        # else:
        #     particle = ''
        prompt = 'Rephrase the question "Are they typically ' + size + '?"'
        prompt += " to use correct grammar, including modification of plurality, particle, or gender. Do not change any of the key words or concepts. "
        prompt += "The question should be a simple yes or no question. Respond with the question only without any additional introduction or conclusion."
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def fix_grammar(self, prompt, temp=1.4, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'Rephrase the following question to use correct grammar: ' + prompt
        prompt += ' Do not change any of the key words or concepts. The question should be a simple yes or no question. Respond with the question only without any additional introduction or conclusion.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def part_of_speech(self, prompt, temp=0.1, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'What is the English grammar part of speech of the word "' + prompt + '"? '
        prompt += 'Respond with the single word part of speech only. For example, repond with one of the following: noun, adjective, adverb, preposition.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def negative_category(self, category, tested_category, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = 'Create a simple, singular category label for ' + category 
        prompt += ' that are NOT ' + tested_category + ' which includes all other possible ' + category
        prompt += '. Use common phrasing. Respond with a single label only, with no introductory or concluding text.'
        return self.run_prompts(messages, prompt, temp, DEBUG)
        
    # Thirty diverse things question maker
    def question_thirty(self, prior_output, questions=None, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        
        chat_template = "Create a simple yes-or-no question, responding with the question only "
        chat_template += "and no introduction or additional verbose details. "
        chat_template += "The question should broadly categorize and divide the following into two equally-sized lists: "
        chat_template += prior_output + ".\n\n"
        chat_template += "Do not include questions similar, equivalent, or directly opposite to the following: "
        for q in questions:
            chat_template += q + ", "
        chat_template = chat_template[:-2] + ". "
        chat_template += "Ensure that the question is simple, unambiguous, clear, and can be answered either yes or no. "
        chat_template += "Do not create compound questions. "
        chat_template += "The question may explore different aspects or characteristics of the list items including size, appearance, function, location, usage, and other defining characteristics. "
        chat_template += "The question should create a general or broad classification of the two categories and should not be overly specific."

        return self.run_prompts(messages, chat_template, temp, DEBUG)

    # Do thirty questions, but using geography
    def question_thirty_geo(self, type, plural, prior_output, questions=None, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()

        chat_template = "Create a simple positive yes-or-no question, responding with the question only "
        chat_template += "and no introduction or additional verbose details. "
        chat_template += "The question should broadly categorize and divide the following " + plural + " into two equally-sized lists: "
        chat_template += prior_output + ".\n\n"
        chat_template += "Do not include questions similar, equivalent, or directly opposite to the following: "
        for q in questions:
            chat_template += q + ", "
        chat_template = chat_template[:-2] + ". "
        chat_template += "Ensure that the question is simple, unambiguous, clear, positively-framed, and can be answered either yes or no. "
        chat_template += "Do not create compound questions. "
        chat_template += "The question may explore different aspects or characteristics of the " + plural + " including geographic location by "
        chat_template += "continent, hemisphere, or latitude, population, primary language, membership in international organizations, "
        chat_template += "unique features, or other characteristics. "
        chat_template += "The question should create a general or broad classification of the two categories and should not be overly specific."

        return self.run_prompts(messages, chat_template, temp, DEBUG)

    # Rephrase the question with the keyword as subject
    def rephrase_with_kw(self, question, keyword, temp=1.4, DEBUG=False):
        messages = self.sys_rephrase()
        rephrase_prompt = "Rephrase the following question to use '" + keyword 
        rephrase_prompt += "' as the subject of the sentence, adjusting tense, gender, and plurality as needed: \""
        rephrase_prompt += question + "\" Respond with the question only and no additional introduction or other text. "
        rephrase_prompt += "Do not change any important words other than the subject of the sentence. The rest of the sentence context should be unaltered."
        return self.run_prompts(messages, rephrase_prompt, temp, DEBUG)

    def answer_question(self, question, temp=1.4, DEBUG=False):
        messages = self.sys_answerer()
        ask_prompt = f"In layman's terms, {question} Answer this question in the most general, common sense as possible, ignoring any trivial nuances or exceptions. Please limit response to a single word, either yes or no, with no introduction, conclusion, or other verbose details. Do not add punctuation to the response.\n"
        return normalize(self.run_prompts(messages, ask_prompt, temp, DEBUG))

    # def at_location(self, second_last_cat, last_cat, location, temp=1.4, DEBUG=False):
    #     prompt = "Are " + second_last_cat + " that are " + last_cat + " typically found in or at " + location + "?"
    #     return self.answer_question(prompt, temp, DEBUG)

    def singular_plural(self, phrase, temp=0.1, DEBUG=False):
        if phrase is None:
            return '', ''
        messages = self.sys_grammar_editor()
        ask_prompt = "Give me the singular and plural form of the phrase '" + phrase 
        ask_prompt += "' as two entries separated by the word aardvark. List the singular form first, the word aardvark, "
        ask_prompt += "then the plural form. Do not add additional introductory or concluding text."
        output = self.run_prompts(messages, ask_prompt, temp, DEBUG)
        pieces = output.split(' aardvark ')
        if len(pieces) == 2:
            for i in range(0,2):
                if 'aardvark' in pieces[i].lower():
                    pieces[i] = pieces[i].lower().replace('aardvark', '')
            return pieces[0], pieces[1]
        else:
            if 'aardvark' in output:
                output = output.lower().replace('aardvark', '')
            return output, output

    def proper_name(self, keyword, temp=0.1, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = "Is '" + keyword + "' a proper name? Reply yes or no only, with no introduction or extra verbose text. Do not add punctuation to the response."
        return normalize(self.run_prompts(messages, prompt, temp, DEBUG))

    def subject(self, question, temp=0.1, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = "What noun, noun phrase, or pronoun is the subject of the following question?\n"
        prompt += "'" + question + "' Respond with the subject word only with no introduction or other verbose text. Do not add punctuation to the response."
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def plural(self, keyword, temp=0.1, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = "Is '" + keyword + "' in grammatically plural form? Reply yes or no only, with no intoduction or other verbose text. Do not add punctuation to the response."
        return normalize(self.run_prompts(messages, prompt, temp, DEBUG))

    def locations_list(self, second_last_cat, last_cat, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = "Provide several broad categories of locations where " + second_last_cat 
        prompt += " that are " + last_cat + " are most often located. Respond with the category names in comma-separated format only."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def continental_regions_list(self, continent, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = "What are a few commonly-used subdivisions to describe country groups within " + continent 
        prompt += "? Respond with the subdivisions in comma-separated form with no introduction or additional verbose text."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def continental_region_subs_list(self, continent, region, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = "What are a few commonly-used subdivisions to describe country groups within the " + region + " subregion of " + continent 
        prompt += "? Respond with the largest four or five subdivisions in comma-separated form with no introduction or additional verbose text."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def country_subregions_list(self, country, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        prompt = "What are a few commonly-used subdivisions to describe groups of states, provinces, or other areas within " + country 
        prompt += "? Respond with the largest four or five subdivisions in comma-separated form with no introduction or additional verbose text."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def is_a_country(self, to_test, temp=0.6, DEBUG=False):
        messages = self.sys_answerer()
        prompt = "Is " + to_test + " a single, sovereign country? Respond with the single word yes or no only, with no added text, verbosity, or punctuation."
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def country_list(self, continent, region, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        if region is not None:
            prompt = "Enumerate all of the countries in the " + region + " subregion of " + continent 
        else:
            prompt = "Enumerate all of the countries in " + continent 
        prompt += ", ordered from largest to smallest. Respond with the country names in comma-separated form with no introduction or additional verbose text."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def country_list(self, continent, region, subregion, temp=1.4, DEBUG=False):
        messages = self.sys_listmaker()
        if region is not None and subregion is not None:
            prompt = "Enumerate all of the countries in the " +subregion+ " subregion of the " + region + " subregion of " + continent 
        elif region is not None:
            prompt = "Enumerate all of the countries in the " + region + " subregion of " + continent 
        else:
            prompt = "Enumerate all of the countries in " + continent 
        prompt += ". Respond with the country names in comma-separated form with no introduction or additional verbose text."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def reframe_as_statement(self, question, answer, temp=1.4, DEBUG=False):
        messages = self.sys_rephrase()
        prompt = "The answer to: " + question + " is " + answer + ". Restate this question and answer as a single statement, accurately reflecting the substance of both the question and the answer. "
        prompt += "Respond with the statement only, with no introduction or other text added."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def update_summary(self, old_summary, new_statement, temp=1.4, DEBUG=False):
        messages = self.sys_rephrase()
        prompt = "A category is described by the following category summary: " + old_summary
        prompt += " Update this detailed summary to reflect the following new information, consolidating redundant information where needed: " + new_statement
        prompt += " If the new information conflicts with the summary, assume the new information is correct. Do not remove any significant details such as abjectives or exclusions. Respond with the revised, detailed summary text only. Do not add any introduction or additional verbosity."
        output = self.run_prompts(messages, prompt, temp, DEBUG)
        return output

    def alpha_check(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'Does the question "' + question + '" ask whether the keyword comes after another word in alphabetical order, sorting order, and/or lexicographical order? Answer yes or no only with no introduction or additional text. Do not add punctuation to the response.'
        return normalize(self.run_prompts(messages, prompt, temp, DEBUG))

    def alpha_extract_word(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'What word or letter does the question "'+question+'" want me to compare the keyword to? Respond with the noun phrase, word, or letter only. Do not add punctuation to the response.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def alpha_earlier_later(self, question, test_word, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'Does the question "'+question+'" want me to test whether the keyword is earlier or later compared to "'+test_word+'" in lexicographical order? Respond with the word earlier or later only. Do not add punctuation to the response.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def alpha_container_check(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'Does the question "' + question + ' explicitly list a particular letter or list of letters? Answer yes or no only, with no additional text. Do not add punctuation to the response.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def alpha_extract_letters(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'What letters does the question "' + question + '" want me to test? Respond with the individual letter or letters only in a comma-separated list, with no additional text.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def alpha_begins_contains(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'Does the question "'+question+'" want me to test whether the keyword begins with or contains particular letters? Respond with either the single word "begins" or "contains" only, and do not add any additional text or introduction. Do not add punctuation to the answer.'
        return self.run_prompts(messages, prompt, temp, DEBUG)

    def alpha_explicit_list(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'Does the question "'+question+'" include an explicit list of words or phrases to match an unknown keyword to? Respond with yes or no only. Do not add punctuation to the answer.'
        return normalize(self.run_prompts(messages, prompt, temp, DEBUG))

    def alpha_extract_list(self, question, temp=0.6, DEBUG=False):
        messages = self.sys_grammar_editor()
        prompt = 'What are the entries in the explicit list provided by the question "'+question+'" Respond with the list entries in comma-separated format only with no additional text.'
        return self.run_prompts(messages, prompt, temp, DEBUG)


        



        
    