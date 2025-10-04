import nn

if __name__ == "__main__":
    model = nn.create_and_compile_model()
    nn.train(model, "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.3,0.7")
    print(nn.predict(model, "0.1,0.2,0.3,0.4,0.5,0.6,0.7"))
