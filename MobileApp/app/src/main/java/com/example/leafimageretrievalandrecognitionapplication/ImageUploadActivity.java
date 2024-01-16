package com.example.leafimageretrievalandrecognitionapplication;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;
import com.google.firebase.firestore.FirebaseFirestore;
import java.util.HashMap;
import java.util.Map;
import java.io.ByteArrayOutputStream;

import androidx.appcompat.app.AppCompatActivity;

public class ImageUploadActivity extends AppCompatActivity {

    private Bitmap selectedImageBitmap;
    private EditText editTextSpecies;
    private EditText editTextDescription;
    private FirebaseFirestore firestore;


    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_upload);

        // Retrieve the image bitmap from the ImageData class
        selectedImageBitmap = ImageData.getImage();

        // Display the selected image in an ImageView
        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageBitmap(selectedImageBitmap);

        // Initialize the EditText fields
        editTextSpecies = findViewById(R.id.editTextSpecies);
        editTextDescription = findViewById(R.id.editTextDescription);

        // Initialize FirebaseFirestore
        firestore = FirebaseFirestore.getInstance();

        // Handle the "Upload" button click
        Button btnUpload = findViewById(R.id.btnUpload);
        btnUpload.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                selectedImageBitmap = Bitmap.createScaledBitmap(selectedImageBitmap, 400, 300, true);
                // Get the entered species and description
                String species = editTextSpecies.getText().toString().trim();
                String description = editTextDescription.getText().toString().trim();
                String selectedBase64Image = convertBitmapToBase64(selectedImageBitmap);

                // Perform validation and handle the upload process
                if (species.isEmpty() || description.isEmpty()) {
                    Toast.makeText(ImageUploadActivity.this, "Please enter species and description", Toast.LENGTH_SHORT).show();
                } else {
                    // Create a map with the image, species, and description
                    Map<String, Object> data = new HashMap<>();
                    data.put("Image", selectedBase64Image); // Placeholder for the image
                    data.put("Species", species);
                    data.put("Description", description);

                    uploadDataToFirestore(data);
                    // Show a success message
                    Toast.makeText(ImageUploadActivity.this, "Image uploaded successfully", Toast.LENGTH_SHORT).show();
                    // Finish the activity
                    finish();
                }
            }
        });

        // Handle the "back" icon click
        ImageView backIcon = findViewById(R.id.backIcon);
        backIcon.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                onBackPressed();
            }
        });
    }

    private void uploadDataToFirestore(Map<String, Object> data) {
        // Add a new document to the "NewLeafPending" collection
        firestore.collection("NewLeafPending")
                .add(data)
                .addOnSuccessListener(documentReference -> {
                    // Show a success message
                    Toast.makeText(ImageUploadActivity.this, "Image uploaded successfully", Toast.LENGTH_SHORT).show();

                    // Finish the activity
                    finish();
                })
                .addOnFailureListener(e -> {
                    // Show an error message
                    Toast.makeText(ImageUploadActivity.this, "Failed to upload image", Toast.LENGTH_SHORT).show();
                    e.printStackTrace();
                });
    }

    private String convertBitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
        byte[] imageBytes = baos.toByteArray();
        return Base64.encodeToString(imageBytes, Base64.DEFAULT);
    }
}
