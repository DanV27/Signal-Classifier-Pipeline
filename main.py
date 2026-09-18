from datasets import Dataset, load_dataset, Audio, load_from_disk


def run():
    ds = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    small = ds.take(200)

    small_ds = Dataset.from_list(list(small))
    small_ds.save_to_disk("data/peoples_speech_sample")

def load():
    ds = load_from_disk("data/peoples_speech_sample")

    print(type(ds))
    print(ds)



if __name__ == "__main__":
    #run()

    load()
