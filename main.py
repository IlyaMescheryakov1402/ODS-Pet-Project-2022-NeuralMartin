from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.dispatcher.filters import Text
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
import re
from neuroMartin import generate_phrase, generate_paragraph

token = '5259939935:AAGidM885StxG8clxnMZNJGo0ijUPDoCKao'
bot = Bot(token=token)
dp = Dispatcher(bot)


# from aiogram import types
@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await message.answer('Сначала мне надо настроиться на работу.. Прошу не ругать и немного подождать')
    global model
    # device = 'cpu'
    print('device - cuda' if torch.cuda.is_available() else 'device - cpu')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    path_to_model = Path('kaggle/input/model-snapshot/fantasy-10.pt')
    global dataset_tokenizer
    dataset_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset_tokenizer.pad_token = dataset_tokenizer.eos_token
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['абзац', 'фразу']
    keyboard.add(*buttons)
    await message.answer("Отлично! Поехали! Генерируем абзац или фразу?", reply_markup=keyboard)

@dp.message_handler(lambda message: message.text == "фразу")
async def without_puree(message: types.Message):
    print('выводить фразу')
    await message.reply("Отличный выбор!", reply_markup=types.ReplyKeyboardRemove())
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['очень маленькую', 'маленькую', 'среднюю', 'большую']
    keyboard.add(*buttons)
    await message.answer("Какого размера фразу?", reply_markup=keyboard)

@dp.message_handler(lambda message: message.text in ['очень маленькую', 'маленькую', 'среднюю', 'большую'])
async def without_puree(message: types.Message):
    await message.reply("Отлично! Теперь напиши мне начало фразу, чтобы я мог её продолжить", reply_markup=types.ReplyKeyboardRemove())
    dct_size = {'очень маленькую': 'extrasmall', 'маленькую': 'small', 'среднюю': 'middle', 'большую': 'big'}
    size_type = dct_size[message.text]
    print('выводить фразу размера ' + size_type)

    @dp.message_handler(content_types=["text"])
    async def handle_text(message):
        await message.reply('Так..теперь мне надо подумать...\n')
        global dataset_tokenizer
        global model
        generate_text = generate_phrase(message.text, model, dataset_tokenizer, size_type)
        print('Полученный текст: ' + generate_text)
        return await message.reply(generate_text)

@dp.message_handler(lambda message: message.text == "абзац")
async def without_puree(message: types.Message):
    await message.reply("Так..теперь мне надо подумать...\n", reply_markup=types.ReplyKeyboardRemove())
    global dataset_tokenizer
    global model
    generate_text = generate_paragraph(model, dataset_tokenizer)
    return await message.reply(generate_text)

@dp.message_handler(commands=["help"])
async def send_welcome(message: types.Message):
    await message.reply('Пишем ветра зимы вместе с нейросетью. Для начала введите команду start')


@dp.message_handler(content_types=['sticker'])
async def handle_sticker(message):
    await message.reply("Nice sticker bro :)")

if __name__ == '__main__':
    executor.start_polling(dp)

