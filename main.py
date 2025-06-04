from custom_srgan import train, run

def main():
    #train.train()
    input_img = "test_images/sample.png"
    checkpoint = "checkpoints/generator.pth"
    output_img = "results/upscaled_sample.png"

    run.upscale(input_img, checkpoint, output_img)

if __name__ == "__main__":
    main()