package com.example.leafimageretrievalandrecognitionapplication;

import android.graphics.Bitmap;

public class ImageData {
    private static Bitmap image;

    public static void setImage(Bitmap bitmap) {
        image = bitmap;
    }

    public static Bitmap getImage() {
        return image;
    }
}