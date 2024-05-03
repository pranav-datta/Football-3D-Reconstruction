using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json; // Import Newtonsoft.Json namespace



public class PlayerController : MonoBehaviour
{
    public GameObject playerObj; 
    public string playerIndex;
    private float[][][] homographyData; //load in from homography
    Dictionary<string, List<List<int>>> jsonData;
    private int numObjects = 22;
    private float[][][] dummyData = new float[100][][]; // Assuming 100 frames and each frame has a variable number of objects
    private int frameNumber = 0;


    // Start is called before the first frame update
    void Start()
    {
        ReadInData();
        // FillDummyData();
    }

    // Update is called once per frame
    void Update()
    {
        if(frameNumber < 179) // Ensure frameNumber is within bounds
        {
            // playerObj.transform.position = new Vector3(dummyData[frameNumber][playerIndex][0], 2, dummyData[frameNumber][playerIndex][1]);
            playerObj.transform.position = new Vector3(jsonData[playerIndex][frameNumber][1], 0.5f, jsonData[playerIndex][frameNumber][2]);
            frameNumber++;
        }
    }

    void ReadInData() 
    {
        string filePath = Path.Combine(Application.dataPath, "updated_player_positions.json");
        string jsonString = File.ReadAllText(filePath);
        // Debug.Log(jsonString);
        jsonData = JsonConvert.DeserializeObject<Dictionary<string, List<List<int>>>>(jsonString);
        // Debug.Log(jsonData.Count);
        
    }

    void FillDummyData()
    {
        // Fill the array with dummy data
        // for (int frame = 0; frame < 100; frame++)
        // {
        //     dummyData[frame] = new float[numObjects][]; // Initialize array for objects in this frame
        //     for (int obj = 0; obj < numObjects; obj++)
        //     {
        //         // Generate random x, y locations for each object
        //         float x = Random.Range(0f, 10f); // Example range: 0 to 10
        //         float y = Random.Range(0f, 10f); // Example range: 0 to 10
        //         // Store x, y locations in the array
        //         dummyData[frame][obj] = new float[] { x, y };
        //     }
        // }
    }
}
