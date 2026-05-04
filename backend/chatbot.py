"""
Emotional State Chatbot Module
Conducts multi-phase conversation for emotional state assessment and influence study.
"""

import random
import re
import torch
from typing import Dict, List, Optional, Tuple

# Stop signal constant
STOP_SIGNAL = "##STUDY_STOPPED##"


class EmotionalStateChatbot:
    """
    A chatbot that conducts emotional state research through conversation.
    
    Phases:
    1. Casual conversation (8 turns)
    2. Standardized assessment (DASS-21 & PANAS)
    3. Influence phase (5 turns with news stories)
    4. Re-assessment (PANAS only)
    """
    
    def __init__(self, model, tokenizer, collection, 
                 happy_news_stories: List[Dict], 
                 sad_news_stories: List[Dict]):
        """
        Initialize the chatbot.
        
        Args:
            model: PyTorch language model
            tokenizer: Tokenizer for the model
            collection: ChromaDB collection with assessment questions
            happy_news_stories: List of positive news stories
            sad_news_stories: List of negative news stories
        """
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.happy_news_stories = happy_news_stories
        self.sad_news_stories = sad_news_stories

        # Conversation state
        self.conversation_history = []
        self.all_user_inputs = []
        self.turns_count = 0

        # Assessment tracking
        self.questions_asked = {}
        self.selected_questions = []
        self.min_casual_turns = 8
        self.assessment_started = False
        self.current_question_id = None
        self.current_question_text = None
        self.current_question_scale = None
        self.scale_instructions_given = False
        self.panas_instructions_given = False

        # Profile tracking
        self.profile_generated = False
        self.baseline_scores = {}

        # Influence phase tracking
        self.influence_started = False
        self.reassessment_started = False
        self.influence_turn_count = 0
        self.max_influence_turns = 5
        self.baseline_state = None
        self.influence_direction = None
        self.user_situation = ""
        self.influence_intensity = "moderate-strong"
        self.current_influence_story = None

        # Re-assessment tracking
        self.reassessment_scores = {}
        self.reassessment_questions = []
        self.reassessment_instructions_given = False

        # DASS-21 question IDs by subscale
        self.dass_depression = ['DASS_D_3', 'DASS_D_5', 'DASS_D_10', 'DASS_D_13',
                                'DASS_D_16', 'DASS_D_17', 'DASS_D_21']
        self.dass_anxiety = ['DASS_A_2', 'DASS_A_4', 'DASS_A_7', 'DASS_A_9',
                             'DASS_A_15', 'DASS_A_19', 'DASS_A_20']
        self.dass_stress = ['DASS_S_1', 'DASS_S_6', 'DASS_S_8', 'DASS_S_11',
                            'DASS_S_12', 'DASS_S_14', 'DASS_S_18']
        self.panas_positive = ['PANAS_P_1', 'PANAS_P_3', 'PANAS_P_5', 'PANAS_P_9',
                               'PANAS_P_10', 'PANAS_P_12', 'PANAS_P_14', 'PANAS_P_16',
                               'PANAS_P_17', 'PANAS_P_19']
        self.panas_negative = ['PANAS_N_2', 'PANAS_N_4', 'PANAS_N_6', 'PANAS_N_7',
                               'PANAS_N_8', 'PANAS_N_11', 'PANAS_N_13', 'PANAS_N_15',
                               'PANAS_N_18', 'PANAS_N_20']
        self.all_question_ids = (self.dass_depression + self.dass_anxiety +
                                 self.dass_stress + self.panas_positive + self.panas_negative)
        self.all_panas_ids = self.panas_positive + self.panas_negative

    def start_conversation(self) -> str:
        """
        Start a new conversation and return the opening message.
        
        Returns:
            str: Opening message to the user
        """
        # Reset all state
        self.conversation_history = []
        self.all_user_inputs = []
        self.turns_count = 0
        self.questions_asked = {}
        self.selected_questions = []
        self.assessment_started = False
        self.current_question_id = None
        self.current_question_text = None
        self.current_question_scale = None
        self.scale_instructions_given = False
        self.panas_instructions_given = False
        self.profile_generated = False
        self.baseline_scores = {}
        self.influence_started = False
        self.reassessment_started = False
        self.influence_turn_count = 0
        self.baseline_state = None
        self.influence_direction = None
        self.user_situation = ""
        self.current_influence_story = None
        self.reassessment_scores = {}
        self.reassessment_questions = []
        self.reassessment_instructions_given = False

        starting_message = (
            "Hello! Welcome to this emotional state research study. Before we begin, "
            "please know that your wellbeing comes first — at any point during our "
            "conversation, if you feel uncomfortable or simply want to stop, just type "
            "STOP (in all caps) and the study will end immediately, no questions asked.\n\n"
            "With that said — how are you feeling today? I'm here to chat."
        )
        self.conversation_history.append({'role': 'assistant', 'content': starting_message})
        return starting_message

    def chat(self, user_input: str) -> str:
        """
        Main chat method - processes user input and returns response.
        
        Args:
            user_input: The user's message
            
        Returns:
            str: Bot's response, or STOP_SIGNAL if user requested stop
        """
        # SAFE WORD CHECK
        if user_input.strip() == "STOP":
            response = (
                "The study has been stopped as requested. Thank you for participating — "
                "your wellbeing is the most important thing. If you ever want to talk to "
                "someone, please don't hesitate to reach out to a mental health professional "
                "or someone you trust. Take care of yourself. 💙"
            )
            self.conversation_history.append({'role': 'user', 'content': user_input})
            self.conversation_history.append({'role': 'assistant', 'content': response})
            return STOP_SIGNAL

        self.all_user_inputs.append(user_input)
        self.conversation_history.append({'role': 'user', 'content': user_input})
        self.turns_count += 1

        # PHASE 1: Casual conversation
        if not self.assessment_started and self.turns_count < self.min_casual_turns:
            response = self._generate_casual_response()

        # TRANSITION: Select subscales + move to assessment
        elif not self.assessment_started and self.turns_count >= self.min_casual_turns:
            self._select_questions_via_model()
            self._extract_user_situation()
            self.assessment_started = True
            response = self._transition_to_assessment()

        # PHASE 2: Assessment
        elif self.assessment_started and not self.profile_generated:
            if self.current_question_id:
                self._score_response(user_input)

            if len(self.questions_asked) >= len(self.selected_questions):
                response = self._generate_profile()
                self.profile_generated = True
            else:
                response = self._ask_next_question()

        # TRANSITION: Profile shown → start influence phase
        elif self.profile_generated and not self.influence_started:
            self._determine_baseline_state()
            self.influence_started = True
            response = self._start_influence_phase()

        # PHASE 3: Influence phase
        elif self.influence_started and not self.reassessment_started:
            self.influence_turn_count += 1
            if self.influence_turn_count >= self.max_influence_turns:
                response = self._start_reassessment()
                self.reassessment_started = True
            else:
                response = self._generate_influence_response(user_input)

        # PHASE 4: Re-assessment
        elif self.reassessment_started:
            if self.current_question_id:
                self._score_reassessment(user_input)

            if len(self.reassessment_scores) >= len(self.reassessment_questions):
                response = self._generate_final_comparison()
            else:
                response = self._ask_reassessment_question()

        self.conversation_history.append({'role': 'assistant', 'content': response})
        return response

    def get_session_data(self) -> Dict:
        """
        Export complete session data for database storage.
        
        Returns:
            Dict containing all session information
        """
        return {
            'conversation_history': self.conversation_history,
            'turns_count': self.turns_count,
            'baseline_scores': self.baseline_scores,
            'reassessment_scores': self.reassessment_scores,
            'baseline_state': self.baseline_state,
            'influence_direction': self.influence_direction,
            'influence_story': self.current_influence_story,
            'user_situation': self.user_situation,
            'questions_asked': self.questions_asked,
            'phase': self._get_current_phase()
        }

    def _get_current_phase(self) -> str:
        """Get current conversation phase."""
        if not self.assessment_started:
            return "casual_chat"
        elif not self.profile_generated:
            return "baseline_assessment"
        elif not self.influence_started:
            return "awaiting_influence"
        elif not self.reassessment_started:
            return "influence"
        else:
            return "reassessment"

    # =========================================================================
    # Helper methods (these would contain your actual implementation)
    # I'll add placeholders - you'll need to paste in the full methods
    # =========================================================================
    
    def _generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text using the model."""
        # TODO: Add your actual text generation code here
        pass
    
    def _get_context(self, num_turns: int = 4) -> str:
        """Get recent conversation context."""
        # TODO: Add your context extraction code
        pass
    
    def _generate_casual_response(self) -> str:
        """Generate casual conversation response."""
        # TODO: Add your casual response generation code
        pass
    
    def _select_questions_via_model(self):
        """Use model to select relevant subscales."""
        # TODO: Add your question selection code
        pass
    
    def _extract_user_situation(self):
        """Extract user situation from conversation."""
        # TODO: Add situation extraction code
        pass
    
    # ... (more helper methods to be added)
