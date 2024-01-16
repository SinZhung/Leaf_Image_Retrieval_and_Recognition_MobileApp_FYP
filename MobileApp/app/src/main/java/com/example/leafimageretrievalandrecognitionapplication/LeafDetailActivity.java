package com.example.leafimageretrievalandrecognitionapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class LeafDetailActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leaf_detail);

        // Retrieve the data passed from the previous activity
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            String species = extras.getString("species");
            String description = extras.getString("description");

            // Find views in the layout
            ImageView imageView = findViewById(R.id.imageView);
            TextView speciesTextView = findViewById(R.id.speciesTextView);
            TextView descriptionTextView = findViewById(R.id.descriptionTextView);
            ImageView backIcon = findViewById(R.id.backIcon);

            // Set data to the views
            speciesTextView.setText(species);
            descriptionTextView.setText(description);

            // Get the image from ImageData class
            Bitmap image = ImageData.getImage();
            if (image != null) {
                imageView.setImageBitmap(image);
                imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            }

            // Handle the "back" icon click
            backIcon.setOnClickListener(new View.OnClickListener() {
                public void onClick(View v) {
                    onBackPressed();
                }
            });
        }
    }
}