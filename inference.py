import sys
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer, RobertaTokenizerFast
from transformers.integrations import TensorBoardCallback
import os
import datasets
import torch
import numpy as np
from itertools import groupby
from nltk.tokenize import word_tokenize, sent_tokenize
import json

frame_list = ['_', 'Event_instance', 'Luck', 'Medical_professionals', 'Process_stop', 'Agriculture', 'Sleep', 'Request', 'Come_down_with', 'Manufacturing', 'Ingredients', 'Processing_materials', 'Intoxicants', 'Scope', 'Sociability', 'Prison', 'Trust', 'Intercepting', 'Take_place_of', 'Bungling', 'Leadership', 'Presence', 'People_by_morality', 'Activity_prepare', 'Political_locales', 'Cause_harm', 'Expressing_publicly', 'Communication_noise', 'Origin', 'Opportunity', 'Objective_influence', 'Amassing', 'Margin_of_resolution', 'Cause_to_wake', 'Economy', 'Capacity', 'Becoming', 'Forgiveness', 'Be_in_agreement_on_action', 'Amounting_to', 'Departing', 'Defending', 'Deserving', 'Detaining', 'Communication_manner', 'Medical_specialties', 'Evidence', 'People_along_political_spectrum', 'Relational_quantity', 'Transition_to_a_quality', 'Adopt_selection', 'Reparation', 'Amalgamation', 'Inhibit_movement', 'Electricity', 'Topic', 'Relative_time', 'Sending', 'Activity_pause', 'Military', 'Expectation', 'Extradition', 'Aging', 'Examination', 'Have_associated', 'Ingest_substance', 'Preventing_or_letting', 'Extreme_value', 'Achieving_first', 'Usefulness', 'Education_teaching', 'Becoming_silent', 'Finish_competition', 'Accomplishment', 'Cause_to_make_noise', 'Fall_asleep', 'Food', 'Sign', 'Telling', 'Used_up', 'Undergoing', 'Colonization', 'Obscurity', 'Apply_heat', 'Being_in_captivity', 'Eclipse', 'Killing', 'Judgment_direct_address', 'Bail_decision', 'Manner', 'Predicament', 'Delivery', 'Distributed_position', 'Robbery', 'Motion', 'Turning_out', 'Location_in_time', 'Becoming_dry', 'Cause_to_perceive', 'Expected_location_of_person', 'Process_end', 'Dominate_competitor', 'Change_of_phase', 'Being_in_control', 'Halt', 'Left_to_do', 'Project', 'Cause_impact', 'Individual_history', 'Position_on_a_scale', 'Needing', 'Destroying', 'Possession', 'Connecting_architecture', 'Thriving', 'Gathering_up', 'Piracy', 'Shapes', 'Rescuing', 'Perception_active', 'Stage_of_progress', 'Withdraw_from_participation', 'Accompaniment', 'Improvement_or_decline', 'Punctual_perception', 'Becoming_aware', 'Rewards_and_punishments', 'Using', 'Text', 'Being_in_effect', 'Law', 'Kinship', 'Change_event_duration', 'Popularity', 'Infrastructure', 'Arrest', 'Part_whole', 'Regard', 'Sound_level', 'Delimitation_of_diversity', 'Being_at_risk', 'Change_post-state', 'Practice', 'Endangering', 'Earnings_and_losses', 'Body_parts', 'Cogitation', 'Contrition', 'Relation', 'Taking_time', 'Ineffability', 'Process_start', 'Similarity', 'Means', 'Temporal_subregion', 'Range', 'System_complexity', 'Reveal_secret', 'Daring', 'Part_ordered_segments', 'Likelihood', 'Committing_crime', 'Exemplar', 'Memory', 'Fullness', 'Dominate_situation', 'Duration_relation', 'Making_arrangements', 'Scarcity', 'Deciding', 'Being_in_operation', 'Catching_fire', 'Competition', 'Coming_to_be', 'Boundary', 'Typicality', 'Activity_stop', 'Wealthiness', 'Intentionally_create', 'Serving_in_capacity', 'Subjective_influence', 'Being_wet', 'Buildings', 'Commerce_sell', 'Interior_profile_relation', 'Being_dry', 'Besieging', 'Indigenous_origin', 'Progression', 'State_of_entity', 'Architectural_part', 'Intentional_traversing', 'Being_necessary', 'Change_of_leadership', 'Ammunition', 'Suitability', 'Change_operational_state', 'Abounding_with', 'Information', 'Body_movement', 'Inclination', 'Part_piece', 'Supply', 'Point_of_dispute', 'Ride_vehicle', 'Instance', 'Quitting_a_place', 'Invading', 'Candidness', 'Making_faces', 'Encoding', 'Sent_items', 'Religious_belief', 'Fastener', 'Taking_sides', 'Fairness_evaluation', 'Assessing', 'Scouring', 'Sign_agreement', 'Soaking_up', 'Create_physical_artwork', 'Social_event', 'Hearsay', 'Adjusting', 'Enforcing', 'Tolerating', 'Offshoot', 'Social_interaction_evaluation', 'Run_risk', 'People', 'Create_representation', 'Operational_testing', 'Sounds', 'Biological_area', 'Heralding', 'Labor_product', 'Cause_change_of_position_on_a_scale', 'Judgment', 'Undergo_change', 'Cause_to_make_progress', 'Being_employed', 'Speak_on_topic', 'Bearing_arms', 'Documents', 'Catastrophe', 'Categorization', 'Disembarking', 'Evaluative_comparison', 'Medical_conditions', 'Judicial_body', 'Ranked_expectation', 'Success_or_failure', 'Cause_to_amalgamate', 'Rite', 'Hit_or_miss', 'Social_connection', 'Alliance', 'Measure_linear_extent', 'Ground_up', 'Certainty', 'Transfer', 'Hiring', 'Clothing', 'Cause_motion', 'Being_attached', 'Out_of_existence', 'Reliance', 'Capability', 'Memorization', 'Storing', 'Offenses', 'Misdeed', 'Communication_response', 'First_experience', 'Fear', 'Just_found_out', 'Foreign_or_domestic_country', 'Scrutiny', 'Respond_to_proposal', 'Mass_motion', 'Going_back_on_a_commitment', 'Rate_description', 'Breathing', 'Desiring', 'Using_resource', 'Intentionally_act', 'Cure', 'Having_or_lacking_access', 'Render_nonfunctional', 'Representative', 'Commerce_pay', 'Notification_of_charges', 'Prominence', 'Closure', 'Preference', 'Imprisonment', 'Desirable_event', 'Ratification', 'Getting_vehicle_underway', 'Beyond_compare', 'History', 'Attention', 'Translating', 'Disgraceful_situation', 'Noise_makers', 'Addiction', 'Attending', 'Seeking_to_achieve', 'Measurable_attributes', 'Attaching', 'Gizmo', 'Partiality', 'Adjacency', 'Meet_with', 'Cause_fluidic_motion', 'Institutionalization', 'Motion_noise', 'Make_noise', 'Perception_experience', 'Temperature', 'Criminal_investigation', 'Completeness', 'Waiting', 'Discussion', 'Reassuring', 'Be_in_agreement_on_assessment', 'Vocalizations', 'Undergo_transformation', 'Quantity', 'Quantified_mass', 'Forgoing', 'Obviousness', 'Measure_area', 'Referring_by_name', 'Performing_arts', 'Directional_locative_relation', 'Cutting', 'Physical_artworks', 'Accoutrements', 'Complaining', 'Grinding', 'Being_active', 'Estimating', 'Reason', 'Shoot_projectiles', 'Fleeing', 'Cause_expansion', 'Arraignment', 'Abandonment', 'Statement', 'Verdict', 'Text_creation', 'Placing', 'Biological_urge', 'Beat_opponent', 'Activity_start', 'Filling', 'Stimulus_focus', 'Have_as_requirement', 'Make_acquaintance', 'Putting_out_fire', 'Purpose', 'Mental_stimulus_stimulus_focus', 'Offering', 'Agree_or_refuse_to_act', 'Remembering_experience', 'Giving_in', 'Measure_mass', 'Sidereal_appearance', 'Linguistic_meaning', 'Eventive_affecting', 'Process_completed_state', 'Emotions_by_stimulus', 'Prohibiting_or_licensing', 'Measure_duration', 'Experience_bodily_harm', 'Natural_features', 'Emergency_fire', 'Frequency', 'Response', 'Frugality', 'Non-gradable_proximity', 'Estimated_value', 'Animals', 'Front_for', 'Kidnapping', 'Mental_stimulus_exp_focus', 'Measure_volume', 'Work', 'Adducing', 'Exchange', 'Manipulate_into_doing', 'Giving_birth', 'Locale_by_event', 'Evoking', 'Body_mark', 'Shopping', 'Moving_in_place', 'Version_sequence', 'Communicate_categorization', 'Imposing_obligation', 'Proportion', 'Trying_out', 'Bringing', 'Growing_food', 'Mining', 'Participation', 'Path_shape', 'Distinctiveness', 'Rank', 'Board_vehicle', 'Interrupt_process', 'Color_qualities', 'Performers_and_roles', 'Expansion', 'Compliance', 'Execution', 'Successful_action', 'Rotting', 'Judgment_communication', 'Partitive', 'Excreting', 'Expertise', 'Existence', 'Exporting', 'Give_impression', 'Experiencer_obj', 'Activity_resume', 'Affirm_or_deny', 'Co-association', 'Emphasizing', 'Irregular_combatants', 'Legality', 'Money', 'Guilt_or_innocence', 'People_by_residence', 'Tasting', 'Extreme_point', 'Degree_of_processing', 'Cause_to_start', 'Wearing', 'Diversity', 'Historic_event', 'Public_services', 'Setting_fire', 'Cause_change', 'Actually_occurring_entity', 'Isolated_places', 'Member_of_military', 'Temporary_stay', 'Abusing', 'Dispersal', 'Giving', 'Dimension', 'Path_traveled', 'Direction', 'Stinginess', 'Strictness', 'Behind_the_scenes', 'Being_obligated', 'Make_agreement_on_action', 'Change_posture', 'Attack', 'Fields', 'Billing', 'Medium', 'Activity_finish', 'Research', 'Cause_bodily_experience', 'Change_tool', 'Vehicle', 'Emotion_directed', 'Process', 'Nuclear_process', 'Control', 'Level_of_force_resistance', 'Possibility', 'Arson', 'Avoiding', 'Roadways', 'Creating', 'Claim_ownership', 'Active_substance', 'Convey_importance', 'Supporting', 'Separating', 'Labeling', 'Sentencing', 'Attempt_means', 'Light_movement', 'Businesses', 'Cause_to_continue', 'Verification', 'Forging', 'Fluidic_motion', 'Team', 'Grasp', 'Being_relevant', 'Travel', 'Temporal_collocation', 'State_continue', 'Volubility', 'Cause_change_of_phase', 'Remembering_information', 'Opinion', 'Commerce_buy', 'Part_inner_outer', 'Launch_process', 'Destiny', 'Try_defendant', 'Execute_plan', 'Explaining_the_facts', 'People_by_origin', 'Age', 'Confronting_problem', 'Hostile_encounter', 'Assistance', 'Arranging', 'Mental_property', 'Abundance', 'Breaking_out_captive', 'Manner_of_life', 'Hit_target', 'Traversing', 'Employing', 'Emanating', 'Taking', 'Redirecting', 'People_by_vocation', 'People_by_religion', 'Body_description_holistic', 'Timespan', 'Revenge', 'Medical_intervention', 'Appointing', 'Hospitality', 'Commemorative', 'Terrorism', 'Surrendering_possession', 'Choosing', 'Entering_of_plea', 'Come_together', 'Concessive', 'System', 'Building', 'Awareness_status', 'Type', 'Motion_directional', 'Name_conferral', 'Sequence', 'Artificiality', 'Hunting', 'Degree', 'Transition_to_state', 'Prevent_or_allow_possession', 'Pattern', 'Aiming', 'Quitting', 'Retaining', 'Recording', 'Judgment_of_intensity', 'Craft', 'Cardinal_numbers', 'Membership', 'Simple_name', 'Terms_of_agreement', 'Damaging', 'Required_event', 'Source_of_getting', 'Reading_activity', 'Death', 'Secrecy_status', 'Biological_entity', 'Probability', 'Store', 'Institutions', 'Unattributed_information', 'Arriving', 'Size', 'Impression', 'Becoming_a_member', 'Self_motion', 'Cooking_creation', 'Willingness', 'Cause_to_fragment', 'Collaboration', 'Communication', 'Conduct', 'Locale_by_use', 'Cause_emotion', 'Fame', 'Ambient_temperature', 'Locative_relation', 'Gesture', 'Rest', 'Rape', 'Forming_relationships', 'Cause_to_resume', 'Locale_by_ownership', 'Weather', 'Inspecting', 'Installing', 'Attributed_information', 'Indicating', 'Unemployment_rate', 'First_rank', 'Activity_ongoing', 'Attempt_suasion', 'Being_questionable', 'Trial', 'Importing', 'Be_subset_of', 'Cause_to_end', 'Fire_burning', 'Compatibility', 'Activity_done_state', 'Proliferating_in_number', 'Removing', 'Accuracy', 'Emptying', 'Lively_place', 'Reading_perception', 'Part_orientational', 'Aggregate', 'Chatting', 'Spatial_co-location', 'Locale', 'Awareness', 'Commercial_transaction', 'Sole_instance', 'Familiarity', 'Occupy_rank', 'Process_resume', 'Suasion', 'Color', 'Thwarting', 'Organization', 'Coming_to_believe', 'Theft', 'Reference_text', 'Connectors', 'Hindering', 'Omen', 'Containers', 'Preliminaries', 'Sufficiency', 'Facial_expression', 'Morality_evaluation', 'Being_located', 'Justifying', 'Intentionally_affect', 'Deny_or_grant_permission', 'Visiting', 'Legal_rulings', 'Posture', 'Network', 'People_by_jurisdiction', 'Proper_reference', 'Substance', 'Surviving', 'Smuggling', 'Commitment', 'Weapon', 'Suspicion', 'Subversion', 'Sensation', 'Ceasing_to_be', 'Containing', 'Contacting', 'Conquering', 'Importance', 'Submitting_documents', 'Firing', 'Cause_change_of_strength', 'Correctness', 'Exchange_currency', 'Feeling', 'Temporal_pattern', 'Causation', 'Predicting', 'Protecting', 'Preserving', 'Relational_natural_features', 'Releasing', 'Reasoning', 'Residence', 'Replacing', 'Receiving', 'Reshaping', 'Expensiveness', 'Reporting', 'Subordinates_and_superiors', 'Operate_vehicle', 'Manipulation', 'Rebellion', 'Touring', 'Location_of_light', 'Being_operational', 'Remainder', 'Chemical-sense_description', 'Entity', 'Desirability', 'Commerce_scenario', 'Food_gathering', 'Holding_off_on', 'Within_distance', 'Resolve_problem', 'Questioning', 'Being_named', 'Risky_situation', 'Negation', 'Calendric_unit', 'Alternatives', 'Renting', 'Reliance_on_expectation', 'Increment', 'Simple_naming', 'Clothing_parts', 'Simultaneity', 'Rejuvenation', 'Precipitation', 'Renunciation', 'Prevarication', 'Attempt', 'Law_enforcement_agency', 'Ingestion', 'Level_of_force_exertion', 'Inclusion', 'Spatial_contact', 'Custom', 'Hiding_objects', 'People_by_age', 'Contingency', 'Coincidence', 'Impact', 'Quarreling', 'Aesthetics', 'Cognitive_connection', 'Getting', 'Being_incarcerated', 'Coming_up_with', 'Change_event_time', 'Setting_out', 'Openness', 'Assemble', 'Reading_aloud', 'Difficulty', 'Change_position_on_a_scale', 'Planned_trajectory', 'Becoming_separated', 'Cause_to_move_in_place', 'Continued_state_of_affairs', 'Experiencer_focus', 'Seeking', 'Emotions_of_mental_activity', 'Immobilization', 'Firefighting', 'Reforming_a_system', 'Identicality', 'Locating', 'Event', 'Attitude_description', 'Personal_relationship', 'Goal', 'Artifact', 'Emotion_active', 'Recovery', 'Duration_description', 'Speed_description', 'Relational_political_locales', 'Win_prize', 'Rate_quantification', 'Summarizing', 'Cause_to_experience', 'Activity_ready_state', 'Sharpness', 'Escaping', 'Waking_up', 'Toxic_substance', 'Dead_or_alive', 'Differentiation', 'Operating_a_system', 'Change_direction', 'Proportional_quantity', 'Domain', 'Time_vector', 'Ordinal_numbers', 'Trendiness', 'Idiosyncrasy', 'Building_subparts', 'Being_born', 'Being_in_category', 'Process_continue', 'Carry_goods', 'Duplication', 'Make_cognitive_connection', 'Cotheme']

def get_true_label(predictions, pad_mask = None, labels = None, input_ids = None, ignore_index=-100):
    """去掉padding/BPE造成的填充label

    Args:
        pred ([type]): 预测到的label，非logits。可以是1-D，也可以是2-D array
        labels ([type]): 同pred的shape
        ignore_index: 要忽略的label id
    """
    if pad_mask is None:
        if labels is None:
            raise ValueError('pad_mask and labels cannot be both None')
        else:
            pad_mask = labels
    if len(predictions.shape)==2:
        print('&&& Assuming tagging predictions:')
        true_predictions = [
            [p for (p, l) in zip(prediction, pad) if l != ignore_index]
            for prediction, pad in zip(predictions, pad_mask)
        ]
        if labels is not None:
            true_labels = [
                [l for (p, l, d) in zip(prediction, label, pad) if d != ignore_index]
                for prediction, label, pad in zip(predictions, labels, pad_mask)
            ]
        if input_ids is not None:
            true_input_ids = [
                [i for (i, d) in zip(input_id, pad) if d != ignore_index]
                for input_id, pad in zip(input_ids, pad_mask)
            ]
    elif len(predictions.shape)==1:
        true_predictions = [p for p,d in zip(predictions, pad_mask) if d !=ignore_index]
        if labels is not None:
            true_labels = [l for p,l,d in zip(predictions, labels, pad_mask) if d !=ignore_index]
    else:
        raise ValueError('Do not support non 2-d, 1-d inputs')
    
    output = (true_predictions,)
    if labels is not None:
        output += (true_labels,)
    if input_ids is not None:
        output += (true_input_ids,)
    return output

def tokenize_and_alingn_labels(ds, tokenize_col, tagging_cols = {}, max_length=256):
    results={}
    for k,v in ds.items():
        if k != tokenize_col:
            results[k]=v
            continue
        out_=tokenizer(v, is_split_into_words=True, max_length=max_length, truncation=True)
        results.update(out_)
    labels={}
    for i, column in enumerate(tagging_cols.keys()):
        label = ds[column]
        fillin_value = tagging_cols[column]
        words_ids = out_.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in words_ids:
            if word_idx is None:
                label_ids.append(fillin_value)
            elif word_idx!=previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(fillin_value)
            previous_word_idx = word_idx
        labels[column] = label_ids
    
    words_ids = out_.word_ids()
    pad_mask = []
    for word_idx in words_ids:
        if word_idx is None:
            pad_mask.append(-100)
        elif word_idx!=previous_word_idx:
            pad_mask.append(0)
        else:
            pad_mask.append(-100)
        previous_word_idx = word_idx
    
    results.update(labels)
    results['pad_mask'] = pad_mask
    return results

def write_predict_to_file(tokens, metaphor_labels, novel_metaphors, frame_labels, out_file='predictions.csv',):
        
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('Tokens\tBorderline_metaphor\tReal_metaphors\tFrame_label\n')
        for token, mls, novels, fls in zip(tokens, metaphor_labels, novel_metaphors, frame_labels):
            for index_, (t, m, n, f_) in enumerate(zip(token, mls, novels, fls)):
                line = f'{t}\t{m}\t{n}\t{frame_list[f_]}'
                f.write(line+'\n')
            f.write('\n')
    print(f'Save to conll file {out_file}.')
    return

if __name__ == '__main__':

    input_file, = sys.argv[1:]
    assert os.path.exists(input_file), f'Input file {input_file} does not exist.'

    metaphor_model = 'CreativeLang/metaphor_detection_roberta_seq'
    novel_metaphor_model = 'CreativeLang/novel_metaphors'
    frame_model = 'liyucheng/frame_finder'

    prediction_output_file = 'predictions.tsv'

    metaphor_model = RobertaForTokenClassification.from_pretrained(metaphor_model, num_labels=2)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=256, )

    # Load example article or articles (you can easily modify this to load multiple articles)
    with open(input_file, 'r', encoding='utf-8') as f:
        example_article = json.load(f)['articles']

    # First, we split the article into sentences.
    sentences = sent_tokenize(example_article)

    # Then, we tokenize the sentences.
    sentences = [word_tokenize(sent) for sent in sentences]
    ds = datasets.Dataset.from_dict({'tokens': sentences})

    ds = ds.map(tokenize_and_alingn_labels, fn_kwargs={'tokenize_col': 'tokens'})

    trainer = Trainer(
        model=metaphor_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(ds[0])
    pred_out = trainer.predict(ds)
    predictions = np.argmax(pred_out.predictions, axis=-1)
    metaphor_predictions, = get_true_label(predictions, pad_mask=ds['pad_mask'])

    del metaphor_model, trainer

    novel_metaphor_model = RobertaForTokenClassification.from_pretrained(novel_metaphor_model, num_labels=2, type_vocab_size=2)
    trainer = Trainer(
        model=novel_metaphor_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    ds = ds.add_column('metaphor_predictions', metaphor_predictions)
    ds = ds.map(tokenize_and_alingn_labels, fn_kwargs={'tokenize_col': 'tokens', 'tagging_cols': {'metaphor_predictions': 0}})
    ds = ds.rename_column('metaphor_predictions', 'token_type_ids')

    print(ds[0])
    pred_out = trainer.predict(ds)
    predictions = np.argmax(pred_out.predictions, axis=-1)
    novel_metaphors, = get_true_label(predictions, pad_mask=ds['pad_mask'])

    del novel_metaphor_model, trainer

    frame_model = RobertaForTokenClassification.from_pretrained(frame_model)
    trainer = Trainer(
        model=frame_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    ds = ds.remove_columns(['token_type_ids'])
    pred_out = trainer.predict(ds)
    predictions = np.argmax(pred_out.predictions, axis=-1)
    frame_predictions, = get_true_label(predictions, pad_mask=ds['pad_mask'])

    # this `where` is important, cannot remove
    filtered_novels = []
    for i, (metaphors, novels) in enumerate(zip(metaphor_predictions, novel_metaphors)):
        novels = np.where(np.array(metaphors) == 1, novels, 0)
        filtered_novels.append(novels)

    write_predict_to_file(ds['tokens'], metaphor_predictions, filtered_novels, frame_predictions, out_file=prediction_output_file)