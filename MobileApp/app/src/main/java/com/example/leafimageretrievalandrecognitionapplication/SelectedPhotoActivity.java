package com.example.leafimageretrievalandrecognitionapplication;

import android.app.ProgressDialog;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class SelectedPhotoActivity extends AppCompatActivity implements CompoundButton.OnCheckedChangeListener {

    private Uri selectedImageUri;
    private Bitmap selectedImageBitmap;

    private CheckBox checkBoxApproach1;
    private CheckBox checkBoxApproach2;
    private CheckBox checkBoxApproach3;
    private Button btnProceed;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_selected_photo);

        // Retrieve the selected photo URI or the captured image from the intent
        if (getIntent().hasExtra("selectedImageUri")) {
            String selectedImageUriString = getIntent().getStringExtra("selectedImageUri");
            selectedImageUri = Uri.parse(selectedImageUriString);
            selectedImageBitmap = convertUriToBitmap(selectedImageUri);

        } else if (getIntent().hasExtra("capturedImageUri")) {
            String selectedImageUriString = getIntent().getStringExtra("capturedImageUri");
            selectedImageUri = Uri.parse(selectedImageUriString);
            selectedImageBitmap = convertUriToBitmap(selectedImageUri);

        } else {
            // Handle the case when no image is available
            finish();
            return;
        }

        // Display the selected photo in an ImageView
        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageBitmap(selectedImageBitmap);

        // Initialize the checkboxes and set the onCheckedChangeListener
        checkBoxApproach1 = findViewById(R.id.checkBoxApproach1);
        checkBoxApproach2 = findViewById(R.id.checkBoxApproach2);
        checkBoxApproach3 = findViewById(R.id.checkBoxApproach3);

        checkBoxApproach1.setOnCheckedChangeListener(this);
        checkBoxApproach2.setOnCheckedChangeListener(this);
        checkBoxApproach3.setOnCheckedChangeListener(this);

        // Initialize the "proceed" button
        btnProceed = findViewById(R.id.btnProceed);
        btnProceed.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Perform image recognition and retrieval tasks using the selected image
                sendImageToServer(selectedImageBitmap);
            }
        });

        // Initialize the "upload" button
        Button btnUpload = findViewById(R.id.btnUpload);
        btnUpload.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Set the selected image bitmap in the ImageData class
                ImageData.setImage(selectedImageBitmap);

                // Create an intent to open the image upload activity
                Intent intent = new Intent(SelectedPhotoActivity.this, ImageUploadActivity.class);

                // Start the activity
                startActivity(intent);
            }
        });

        // Handle the "back" icon click
        ImageView backIcon = findViewById(R.id.backIcon);
        backIcon.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                onBackPressed();
            }
        });

        // Check checkbox states initially
        checkCheckboxStates();
    }

    private Bitmap convertUriToBitmap(Uri imageUri) {
        try {
            Bitmap originalBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);

            // Retrieve the image orientation from the image metadata
            String[] projection = {MediaStore.Images.ImageColumns.ORIENTATION};
            Cursor cursor = getContentResolver().query(imageUri, projection, null, null, null);
            int orientation = 0;
            if (cursor != null && cursor.moveToFirst()) {
                int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.ImageColumns.ORIENTATION);
                orientation = cursor.getInt(columnIndex);
                cursor.close();
            }

            // Rotate the bitmap based on the orientation value
            Matrix matrix = new Matrix();
            matrix.postRotate(orientation);
            return Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.getWidth(), originalBitmap.getHeight(), matrix, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        // Handle checkbox state changes
        if (isChecked) {
            // Uncheck all other checkboxes
            if (buttonView == checkBoxApproach1) {
                checkBoxApproach2.setChecked(false);
                checkBoxApproach3.setChecked(false);
            } else if (buttonView == checkBoxApproach2) {
                checkBoxApproach1.setChecked(false);
                checkBoxApproach3.setChecked(false);
            } else if (buttonView == checkBoxApproach3) {
                checkBoxApproach1.setChecked(false);
                checkBoxApproach2.setChecked(false);
            }
        }

        // Check checkbox states
        checkCheckboxStates();
    }

    private void checkCheckboxStates() {
        boolean isChecked = checkBoxApproach1.isChecked() || checkBoxApproach2.isChecked()
                || checkBoxApproach3.isChecked();
        btnProceed.setEnabled(isChecked);
    }

    private ProgressDialog progressDialog;

    // Inside centerProgressDialog method
    private void centerProgressDialog() {
        if (progressDialog.getWindow() != null) {
            WindowManager.LayoutParams layoutParams = progressDialog.getWindow().getAttributes();
            layoutParams.gravity = Gravity.CENTER;
            progressDialog.getWindow().setAttributes(layoutParams);
        }
    }
    private void sendImageToServer(Bitmap imageBitmap) {
        try {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    progressDialog = new ProgressDialog(SelectedPhotoActivity.this);
                    progressDialog.setMessage("Processing Image...");
                    progressDialog.setCancelable(false);
                    centerProgressDialog();
                    progressDialog.show();
                }
            });

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);
            byte[] imageData = baos.toByteArray();

            OkHttpClient client = new OkHttpClient();

            // Create the request body with form data
            MediaType mediaType = MediaType.parse("multipart/form-data");
            MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", "image.jpg", RequestBody.create(mediaType, imageData));

            // Check which approach checkboxes are selected and add corresponding parameters to the request
            if (checkBoxApproach1.isChecked()) {
                requestBodyBuilder.addFormDataPart("approach", "1");
            }
            if (checkBoxApproach2.isChecked()) {
                requestBodyBuilder.addFormDataPart("approach", "2");
            }
            if (checkBoxApproach3.isChecked()) {
                requestBodyBuilder.addFormDataPart("approach", "3");
            }

            RequestBody requestBody = requestBodyBuilder.build();

            Request request = new Request.Builder()
                    .url("http://192.168.0.155:5000/process_image")
                    .post(requestBody)
                    .build();

            client.newCall(request).enqueue(new okhttp3.Callback() {
                @Override
                public void onResponse(okhttp3.Call call, Response response) throws IOException {
                    // Handle the response here
                    String responseBody = response.body().string();
                    Log.d("Response", responseBody);

                    try {
                        // Hide the loading spinner dialog
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                progressDialog.dismiss();
                            }
                        });

                        // Convert the response JSON to a JSONObject
                        JSONObject jsonResponse = new JSONObject(responseBody);

                        // Retrieve the 'Indices' and 'Scores' arrays from the response
                        JSONArray indicesArray;
                        JSONArray resultArray;
                        if (jsonResponse.has("Result")) {
                            indicesArray = jsonResponse.getJSONArray("Indices");
                            resultArray = jsonResponse.getJSONArray("Result");
                        } else {
                            indicesArray = new JSONArray();
                            resultArray = new JSONArray(); // Create an empty array if the key is not present
                        }

                        // Pass the indices and scores to the FinalResultActivity
                        Intent intent = new Intent(SelectedPhotoActivity.this, FinalResultActivity.class);
                        intent.putExtra("indices", indicesArray.toString());
                        intent.putExtra("result", resultArray.toString());
                        startActivity(intent);

                    } catch (JSONException e) {
                        // Handle JSON parsing error
                        e.printStackTrace();
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(SelectedPhotoActivity.this, "Failed to parse response", Toast.LENGTH_SHORT).show();
                            }
                        });
                    }
                }

                @Override
                public void onFailure(okhttp3.Call call, IOException e) {
                    e.printStackTrace();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(SelectedPhotoActivity.this, "Failed to process the image", Toast.LENGTH_SHORT).show();
                            progressDialog.dismiss();
                        }
                    });
                }
            });

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
