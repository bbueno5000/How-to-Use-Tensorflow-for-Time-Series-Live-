"""
DOCSTRING
"""
import matplotlib.pyplot as pyplot
import numpy
import tensorflow

class TimeSeries:
    """
    DOCSTRING
    """
    def __init__(self):
        self.batch_size = 5
        self.echo_step = 3
        self.num_batches = self.total_series_length // self.batch_size // self.truncated_backprop_length
        self.num_classes = 2
        self.num_epochs = 100
        self.state_size = 4
        self.total_series_length = 50000
        self.truncated_backprop_length = 15

    def __call__(self):
        data = self.generate_data()
        print(data)
        batchX_placeholder = tensorflow.placeholder(
            tensorflow.float32, [self.batch_size, self.truncated_backprop_length])
        batchY_placeholder = tensorflow.placeholder(
            tensorflow.int32, [self.batch_size, self.truncated_backprop_length])
        init_state = tensorflow.placeholder(tensorflow.float32, [self.batch_size, self.state_size])
        W = tensorflow.Variable(numpy.random.rand(
            self.state_size+1, self.state_size), dtype=tensorflow.float32)
        b = tensorflow.Variable(numpy.zeros((1, self.state_size)), dtype=tensorflow.float32)
        W2 = tensorflow.Variable(numpy.random.rand(
            self.state_size, self.num_classes), dtype=tensorflow.float32)
        b2 = tensorflow.Variable(numpy.zeros((1, self.num_classes)), dtype=tensorflow.float32)
        inputs_series = tensorflow.unpack(batchX_placeholder, axis=1)
        labels_series = tensorflow.unpack(batchY_placeholder, axis=1)
        current_state = init_state
        states_series = list()
        for current_input in inputs_series:
            current_input = tensorflow.reshape(current_input, [self.batch_size, 1])
            input_and_state_concatenated = tensorflow.concat(1, [current_input, current_state])
            next_state = tensorflow.tanh(tensorflow.matmul(input_and_state_concatenated, W) + b)
            states_series.append(next_state)
            current_state = next_state
        logits_series = [tensorflow.matmul(state, W2) + b2 for state in states_series]
        predictions_series = [tensorflow.nn.softmax(logits) for logits in logits_series]
        losses = [tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels) for logits, labels in zip(logits_series,labels_series)]
        total_loss = tensorflow.reduce_mean(losses)
        train_step = tensorflow.train.AdagradOptimizer(0.3).minimize(total_loss)
        with tensorflow.Session() as sess:
            sess.run(tensorflow.initialize_all_variables())
            pyplot.ion()
            pyplot.figure()
            pyplot.show()
            loss_list = list()
            for epoch_idx in range(self.num_epochs):
                x, y = self.generate_data()
                _current_state = numpy.zeros((self.batch_size, self.state_size))
                print("New data, epoch", epoch_idx)
                for batch_idx in range(self.num_batches):
                    start_idx = batch_idx * self.truncated_backprop_length
                    end_idx = start_idx + self.truncated_backprop_length
                    batchX = x[:,start_idx:end_idx]
                    batchY = y[:,start_idx:end_idx]
                    _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                        [total_loss, train_step, current_state, predictions_series],
                        feed_dict={
                            batchX_placeholder: batchX,
                            batchY_placeholder: batchY,
                            init_state:_current_state})
                    loss_list.append(_total_loss)
                    if batch_idx % 100 == 0:
                        print("Step",batch_idx, "Loss", _total_loss)
                        self.plot(loss_list, _predictions_series, batchX, batchY)
        pyplot.ioff()
        pyplot.show()

    def generate_data(self):
        """
        The input is a random binary vector.
        The output will be the  “echo” of the input, shifted echo_step steps to the right.
        """
        x = numpy.array(numpy.random.choice(2, self.total_series_length, p=[0.5, 0.5]))
        y = numpy.roll(x, self.echo_step)
        y[0:self.echo_step] = 0
        x = x.reshape((self.batch_size, -1))
        y = y.reshape((self.batch_size, -1))
        return (x, y)

    def plot(self, loss_list, predictions_series, batchX, batchY):
        """
        DOCSTRING
        """
        pyplot.subplot(2, 3, 1)
        pyplot.cla()
        pyplot.plot(loss_list)
        for batch_series_idx in range(5):
            one_hot_output_series = numpy.array(predictions_series)[:, batch_series_idx, :]
            single_output_series = numpy.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
            pyplot.subplot(2, 3, batch_series_idx + 2)
            pyplot.cla()
            pyplot.axis([0, self.truncated_backprop_length, 0, 2])
            left_offset = range(self.truncated_backprop_length)
            pyplot.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
            pyplot.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
            pyplot.bar(left_offset, single_output_series * 0.3, width=1, color="green")
        pyplot.draw()
        pyplot.pause(0.0001)

if __name__ == '__main__':
    time_series = TimeSeries()
    time_series()
