from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim

tokenizer = RegexpTokenizer(r'\w+')

enstop = get_stop_words('en')

pstemmer = PorterStemmer()

doc = """Factiva
Dow Jones
Developers - How To: Develop Android Apps Using MIT App Inventor
Distributed by Contify.com
2138 words
1 July 2016
Open Source FOR You
ATLINX
English
Copyright 2016. EFY Enterprises Pvt. Ltd.
There is a secret inventor inside each of us. Get your creative juices flowing and go ahead and develop an Android app or two. It is as easy as you think it is. Follow the detailed instructions given in this article, and you will have an Android app up and running in next to no time.

Imagine that you have come up with an idea for an app to address your own requirements, but due to lack of knowledge and information, don't know where to begin. You could contact an Android developer, who would charge you for his services, and you would also risk having your idea being copied or stolen. You may also feel that you can't develop the app yourself as you do not have the required programming and coding skills. But that's not true. Let me tell you that you can develop Android apps on your own without any programming and coding; and in this article, I am going to let you into the secret of how to go about doing that.

An introduction to App Inventor

App Inventor is a tool that will convert your idea into a working application without the need for any prior coding or programming skills. App Inventor is the open source utility developed by Google in 2010 and, currently, it is being maintained by the Massachusetts Institute of Technology (MIT). It allows absolute beginners in computer science to develop Android applications. It provides you with a graphical user interface with all the necessary components required to build Android apps. You just need to drag and drop the components in the block editor. Each block is an independent action, which you need to arrange in a logical sequence in order to result in some action.

App Inventor features

App Inventor is a feature-rich open source Android app development tool. Here are its features.

1. Open source: Being open source, the tool is free for everyone and you don't need to purchase anything. Open source software also gives you the freedom to customise it as per your own requirements.

2. Browser based: App Inventor runs on your browser; hence, you don't need to install anything onto your computer. You just need to log in to your account using your email and password credentials.

3. Stored on the cloud: All your app related projects are stored on Google Cloud; therefore, you need not keep anything on your laptop or computer. Being browser based, it allows you to log in to your account from any device and all your work is in sync with the cloud database.

4. Real-time testing: App Inventor provides a standalone emulator that enables you to view the behaviour of your apps on a virtual machine. Complete your project and see it running on the emulator in real-time.

5. No coding required: As mentioned earlier, it is a tool with a graphical user interface, with all the built-in component blocks and logical blocks. You need to assemble multiple blocks together to result in some behavioural action.

6. Huge developer community: You will meet like-minded developers from across the world. You can post your queries regarding certain topics and these will be answered quickly. The community is very supportive and knowledged.

System requirements

Given below are the system requirements to run App Inventor from your computer or laptop:

1. A valid Google account, as you need to log in using your credentials.

2. A working Internet connection, as you need to log in to the cloud-based browser that's compatible with App Inventor; hence, a working Internet connection is a must.

3. App Inventor is compatible with Google Chrome 29+, Safari 5+ and Firefox 23+.

4. An Android phone to test the final, developed application.

Beginning with App Inventor

Hope you have everything to begin your journey of Android app development with App Inventor. Follow the steps below to make your first project.

1. Open your Google Chrome/Safari/ Firefox browser and open the Google home page.

2. Using the search box, search for App Inventor.

3. Choose the very first link. It will redirect you to the App Inventor project's main page. This page contains all the resources and tutorials related to App Inventor. We will explore it later. For now, click on the Create button on the top right corner.

4. The next page will ask for your Google account credentials. Enter your user name and password that you use for your Gmail application.

5. Click on the Sign in button, and you will successfully reach the App Inventor app development page. If the page asks you to confirm your approval of certain usage or policy terms, agree with them all. It is all safe and is mandatory if you want to move ahead.

6. If all is done correctly, you should see a page similar to what's shown in Figure 5.

7. Congratulations! You have successfully set up all the necessary things and can now develop your first application.

Your first Android application

So far, you know what App Inventor is and how to run it on your computer. Now, I'll tell you how to make your very first Android application and, once again, I am saying it is pretty simple. You will not require any programming or coding knowledge for this.

Log in to your App Inventor account using the steps mentioned above and you will land on the My Projects page.

1. Since you have not developed any projects so far, it will not display any list under My Projects. We will begin our first project, by clicking on the Start New Project button.

2. Let's give a suitable name to the project. Since it is our first project together, let's call it First_Project. Please keep in mind that if your project name has more than one word, then you should use the underscore (_) character to join them as App Inventor doesn't allow using spaces. Click on the OK button to proceed.

3. You will now see something similar to the image shown in Figure 8. This is the main page for the development. Here, we will add various components to our project and then add the respective functionality. Don't worry, I will brief you about the various things you will see on the page. The page you are currently viewing is called the designer because we design the app here.

Although we have named our application/project 'First_Project', nothing has been said about what it is supposed to do. We must always have a clear idea of what the app will do, so let's list that out.

Objective

We want to make an app that will work as a theft alarm. The phone will make the sound of a siren whenever it is moved.

So let's think about what the app should look like.

GUI requirements

For now, there is no special GUI requirement, but rather than keeping the screen blank, it is better to show an image on the phone screen that signifies 'danger'. We need to upload an image and a sound that we want to play.

We will require the following components for this project.

Steps to be taken to create the application

1. Drag and drop the button onto the viewer.

2. Now select the Sound under Media and drop it into the viewer. You will see it placed under non-visible components.

3. Now, from the Sensor category, choose Accelerometer Sensor and drop it on the viewer. Similarly, it will also be listed under non-visible components.

4. Next, select Button1 in the components and look for its properties in the Property pane.

5. You will find Text property with some pre-written text for Button1. You need to erase all this text and make it empty.

6. Now, we want a threatening or 'danger' image on the screen; so why not add it to the button?

7. Under Button properties, you will see Image property. Select it, and you will be asked to upload any photo from your computer. Browse the location, and upload the photo to the server. It will also be seen under the media tag on the same page.

8. In the same way, we need to upload the sound file that we want to play.

9. Now select the Sound1 component and under its properties, set its source to the sound file you have uploaded.

10. When you are done with this, your viewer should look something like what's shown in Figure 11. I have used a different image, apart from which, everything else ought to be the same.

Adding behaviour to the blocks

Now, let's head towards the block editor to define the behaviour, and discuss the actual functionality that we are expecting from our application.

First, when the screen is touched, it should play a sound, i.e., the button we have used should sense this touch.

Second, when the phone is moved from its place, it should play a sound, i.e., the accelerometer sensor should sense this movement.

So let's move on and add these two behaviours, using the block editor. There is a button available right above the Properties pane to switch between the designer and block editor.

Block editor blocks

I have already prepared the blocks for you. All you need to do is drag the relevant blocks from the left side palette and drop them on the viewer. Arrange the blocks exactly the way you see in the image. I will explain what each does and how it is called.

 The first block is for the accelerometer. We have added the accelerometer sensor to our project to sense any movement in the physical position of the device. So as soon as the accelerometer senses the movement, the following two events will occur:

 It will call the sound component to play the sound. The sound will be what has been set as a source in the designer.

 It will vibrate the device for 500 milliseconds. The amount of time you want the device to be vibrated can be modified since it is an editable field.

 The second block is to trigger the events when someone touches the screen. We have added a button whose image property is the image you are seeing on your screen. As soon as it feels the touch, it will play the sound and vibrate too. Now you are done with the block editor. Next, let's move to download and install the app on your phone to check how it is working.

Packaging and testing

To test the app, you need to get it on your phone. First, you have to download the application to your computer and then move it to your phone via Bluetooth or USB cable. I'll tell you how to download it.

On the top row, click on the Build button. It will show you the options to download the APK to your computer.

The progress of the download can be viewed and after being successfully completed, the application will be placed in the download folder of your directory or the preferred location you have set for your downloading.

Now, you need to get this APK file to your mobile phone, either via Bluetooth or via USB cable. Once you have placed the APK file on your SD card, you need to install it. Follow the on-screen instructions to install it. You might get some notification or warning saying 'Install from untrusted source'. Allow this from the settings and after a successful installation, you will see the icon of your application in the menu of your mobile. Here, you will see the default icon that can be changed, and we will tell you how to do so as we move ahead in this course.

I hope your application is working exactly as per the requirements you have set. Now, depending upon your usage requirements you can change various things like the image, sound and behaviour too.

A note to readers: I would like to know more about your interests and ideas on creating Android apps. If you have any specific ideas on which you want me to write, do write back to me via email. I will surely try to help you convert your idea into a working application.

Meghraj Singh Beniwal

The author has a B. Tech in electronics and communication, is a freelance writer and an Android app developer. He is currently working as an automation engineer at Infosys, Pune. He can be contacted at meghrajsingh01@rediffmail.com/ meghrajwithandroid@gmail.com.

EFY Enterprises Pvt. Ltd.

Document ATLINX0020160701ec710005x


IUT to hold "Hour of Code" campaign
206 words
28 June 2016
Uzbekistan Daily
UZBDAY"""

docx = [doc]

texts = []

for i in docx:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    stopped_tokens = [i for i in tokens if not i in enstop]

    stemmed_tokens = [pstemmer.stem(i) for i in stopped_tokens]

    texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)
print(ldamodel.print_topics(num_topics=2, num_words=4))