from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import RMSprop
from data_processor import Processor
import numpy as np

START_TOKEN = 'STARTTOKEN'
STOP_TOKEN = 'STOPTOKEN'

'''An implementation of the sequence-to-sequence process (output text generation based on input text).
A great deal of the code in this class comes directly from the following webpage: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html'''
class Seq2Seq:
    
    def fit(self, hateful_posts, responses, post_tokens, resp_tokens, data_processor, latent_dim=100):
        
        ### TRAINING ###
        
        self.post_tokens = list(post_tokens)
        self.resp_tokens = list(resp_tokens) + [START_TOKEN, STOP_TOKEN]
        self.max_post_len = data_processor.get_max_post_tokens()
        self.max_resp_len = data_processor.get_max_resp_tokens()
        self.post_tokenizer = data_processor.get_post_tokenizer()
        self.resp_tokenizer = data_processor.get_resp_tokenizer()

        decoder_input_data = list(map(lambda resp: START_TOKEN + ' ' + resp, responses))
        decoder_target_data = list(map(lambda resp: resp + ' ' + STOP_TOKEN, responses))

        hateful_posts_one_hot = one_hot(hateful_posts, self.post_tokens, self.max_post_len, self.post_tokenizer)
        decoder_input_data_one_hot = one_hot(decoder_input_data, self.resp_tokens, self.max_resp_len + 1, self.resp_tokenizer) # + 1 because of encoding START_TOKEN at the start
        decoder_target_data_one_hot = one_hot(decoder_target_data, self.resp_tokens, self.max_resp_len + 1, self.resp_tokenizer) # + 1 because of encoding STOP_TOKEN at the end
        
        encoder_inputs = Input(shape=(None, len(self.post_tokens)))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, len(self.resp_tokens)))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(len(self.resp_tokens), activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        opt = RMSprop(learning_rate=0.01) # default learning rate is 0.001
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        model.fit([hateful_posts_one_hot, decoder_input_data_one_hot], decoder_target_data_one_hot, batch_size=10, epochs=40, validation_split=0.2)

        ### PREP FOR TEXT GENERATION ###

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    def generate_response(self, post):
        one_hot_post = one_hot([post], self.post_tokens, self.max_post_len, self.post_tokenizer)
        states_value = self.encoder_model.predict(one_hot_post)

        target_seq = np.zeros((1,1,len(self.resp_tokens)))
        target_seq[0, 0, self.resp_tokens.index(START_TOKEN)] = 1

        stop_condition = False
        response = ""
        num_tokens = 0
        while True:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.resp_tokens[sampled_token_index]
            if sampled_token != STOP_TOKEN: 
                response += sampled_token + ' '
                num_tokens += 1

            if sampled_token == STOP_TOKEN or num_tokens >= self.max_resp_len:
                break
            
            target_seq = np.zeros((1, 1, len(self.resp_tokens)))
            # target_seq = np.append(target_seq, np.zeros((1, 1, len(self.resp_tokens))), 1) # a possible alternative for the previous line, but I don't think it works as well
            target_seq[-1, -1, sampled_token_index] = 1
            states_value = [h,c]
        
        return response

def one_hot(strings, token_list, max_string_tokens, tokenizer):
    one_hots = np.zeros((len(strings), max_string_tokens, len(token_list)), dtype=int)
    for i in range(len(strings)):
        curr_string = strings[i]
        curr_string_tokens = tokenizer(curr_string)
        for j in range(len(curr_string_tokens)):
            curr_token = curr_string_tokens[j]
            token_list_index = token_list.index(curr_token)
            one_hots[i,j,token_list_index] = 1
    return one_hots

def main():
    p = Processor()
    X, feature_names, Y, post_texts, post_tokens, actual_responses, resp_tokens = p.process_files('data/gab.csv', 'data/reddit.csv', stop_after_rows=50, overwrite_output_files=False).values()
    hateful_posts = post_texts[np.nonzero(Y)]
    model = Seq2Seq()
    model.fit(hateful_posts, actual_responses, post_tokens, resp_tokens, p) # TODO: if we end up with good enough predictions on training data, then hold out some testing data
    
    indices_to_test = np.random.choice(hateful_posts.shape[0], size=10, replace=False) # TODO: if we end up with good enough predictions on training data, then use testing data here instead
    print("\n\nResponse generation tests:")
    for i in indices_to_test:
        print('\nResponse for "%s"' % hateful_posts[i])
        print('Generated: "%s"' % model.generate_response(hateful_posts[i]))
        print('Actual: "%s"' % actual_responses[i])

def test():
    '''Make sure basic features of the model are working, using a tiny, simple dataset'''
    input_arr = np.array(["a b b c","b d e a","c e a a","d","e e","a a b a",'b a','a c b c','a a a','b b b a'])
    output_arr = np.array(["b c c d","c e f b","d f b b","e","f f",'b b c b','c b','b d c d','b b b','c c c b'])
    input_toks = np.array(["a","b","c","d","e"])
    output_toks = np.array(["b","c","d","e","f"])

    p = Processor()
    p.max_post_tokens = max(map(lambda string: len(string), input_arr))
    p.max_resp_tokens = max(map(lambda string: len(string), output_arr))
    p.post_tokenizer = lambda string: string.split()
    p.resp_tokenizer = lambda string: string.split()

    model = Seq2Seq()
    model.fit(input_arr, output_arr, input_toks, output_toks, p)

    inputs_to_test = ['a','b b','a b a']
    outputs_to_test = ['b','c c','b c b']
    print("\n\nResponse generation tests:")
    for input_str, output_str in zip(inputs_to_test, outputs_to_test):
        print('\nResponse for "%s"' % input_str)
        print('Generated: "%s"' % model.generate_response(input_str))
        print('Actual: "%s"' % output_str)

if __name__ == "__main__":
    main()
