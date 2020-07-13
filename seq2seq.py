from keras.models import Model
from keras.layers import Input, LSTM, Dense
from data_processor import Processor
import numpy as np

START_CHAR = '{'
STOP_CHAR = '}'

class Seq2Seq:
    
    def fit(self, hateful_posts, responses, data_processor):
        
        ### TRAINING ###
        
        self.post_chars = data_processor.get_post_chars()
        self.resp_chars = data_processor.get_resp_chars() + [START_CHAR, STOP_CHAR]
        self.max_post_len = data_processor.get_max_post_len()
        self.max_resp_len = data_processor.get_max_resp_len()

        decoder_input_data = list(map(lambda resp: START_CHAR + resp, responses))
        decoder_target_data = list(map(lambda resp: resp + STOP_CHAR, responses))

        hateful_posts_one_hot = one_hot(hateful_posts, self.post_chars, self.max_post_len)
        decoder_input_data_one_hot = one_hot(decoder_input_data, self.resp_chars, self.max_resp_len + 1) # + 1 because of encoding START_CHAR at the start
        decoder_target_data_one_hot = one_hot(decoder_target_data, self.resp_chars, self.max_resp_len + 1) # + 1 because of encoding STOP_CHAR at the end
        
        encoder_inputs = Input(shape=(None, len(self.post_chars)))
        encoder = LSTM(4, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, len(self.resp_chars)))
        decoder_lstm = LSTM(4, return_sequences=True, return_state=True,)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(len(self.resp_chars), activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([hateful_posts_one_hot, decoder_input_data_one_hot], decoder_target_data_one_hot, batch_size=10, epochs=10, validation_split=0.2)

        ### PREP FOR TEXT GENERATION ###

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(4,))
        decoder_state_input_c = Input(shape=(4,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    def generate_response(self, post):
        one_hot_post = one_hot([post], self.post_chars, self.max_post_len)
        states_value = self.encoder_model.predict(one_hot_post)

        target_seq = np.zeros((1,1,len(self.resp_chars)))
        target_seq[0, 0, self.resp_chars.index(START_CHAR)] = 1

        stop_condition = False
        response = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.resp_chars[sampled_token_index]
            response += sampled_char

            if sampled_char == STOP_CHAR or len(response) > self.max_resp_len:
                stop_condition = True
            
            target_seq = np.zeros((1, 1, len(self.resp_chars)))
            target_seq[0, 0, sampled_token_index] = 1
            states_value = [h,c]
        
        return response

def one_hot(strings, token_list, max_string_len):
    one_hots = np.zeros((len(strings), max_string_len, len(token_list)), dtype=int)
    for i in range(len(strings)):
        curr_string = strings[i]
        for j in range(len(curr_string)):
            curr_char = curr_string[j]
            token_list_index = token_list.index(curr_char)
            one_hots[i,j,token_list_index] = 1
    return one_hots

def main():
    p = Processor()
    gab_X, gab_feature_names, gab_Y, gab_resps = p.process_files('data/gab.csv', stop_after_rows=15)
    hateful_posts = p.get_posts_list()[np.nonzero(gab_Y)]
    model = Seq2Seq()
    model.fit(hateful_posts, gab_resps, p)
    print(model.generate_response(hateful_posts[0]))

if __name__ == "__main__":
    main()
