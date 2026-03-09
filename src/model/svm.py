
class svm_model():
    # 7 SVM
    def train_svm(self, epochs=100):

        print("Starting Training of the SVM Model...")
        
        # At this point self has Pandas Object, convert dataframe to numpy array
        x_input = torch.tensor(self.train_features.values, dtype=torch.float32)
        y_output = torch.tensor(self.train_labels.values, dtype=torch.float32)

        # SVM requirement convert labels 0 -> -1
        y_output = torch.where(y_output == 0, -1, 1)
        
        #define model object with SVM contruction of the
        self.svm_model = SVM(x_input.shape[1])

        # Now we start our Stochastic Gradient Descent Engine
        optimizer = optim.SGD(self.svm_model.parameters(), lr=0.01)
        
        # Start Loop for Datapasses
        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.svm_model(x_input).squeeze()
            # calculate Hinge Weight update
            loss = torch.mean(torch.clamp(1 - y_output * outputs, min=0))
            # Do backwards propagation to calculate Gradient
            loss.backward()
            # Step based on Gradient cal
            optimizer.step()

        print("SVM Training Complete")

    def test_svm(self):

        print("Testing SVM Model...")
        """
        # Cast Datafram to nummpy array
        x_input = torch.tensor(self.test_features.values, dtype=torch.float32)
        
        outputs = self.svm_model(x_input).detach().numpy()

        self.predictions = (outputs >= 0).astype(int).flatten()
        """
        print("Testing Complete")


class SVM(nn.Module):

    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

